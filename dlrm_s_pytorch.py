from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import builtins
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import dlrm_data_pytorch as dp
import optim.rwsadagrad as RowWiseSparseAdagrad

from torch.utils.tensorboard import SummaryWriter

from quantizations import FP8QuantDequantSTE, FP16QuantDequantSTE
from selection_networks import EdimQbitSelectionNetwork

exc = getattr(builtins, "IOError", "FileNotFoundError")


def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, ndevices=1):
    if use_gpu:  # .cuda()
        # lS_i can be either a list of tensors or a stacked tensor.
        # Handle each case below:
        if ndevices == 1:
            lS_i = (
                [S_i.to(device) for S_i in lS_i]
                if isinstance(lS_i, list)
                else lS_i.to(device)
            )
            lS_o = (
                [S_o.to(device) for S_o in lS_o]
                if isinstance(lS_o, list)
                else lS_o.to(device)
            )
    return dlrm(X.to(device), lS_o, lS_i)


def loss_fn_wrap(Z, T, device):
    if args.loss_function == "mse" or args.loss_function == "bce":
        return dlrm.loss_fn(Z, T.to(device))


def unpack_batch(b):
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class DLRM_Net(nn.Module):
    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        loss_function="bce",
        dyedims_flag=False,
        dyedims_options=None,
        dyqbits_flag=False,
        dyqbits_options=None,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
        ):
            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            self.dyedims_flag = dyedims_flag
            self.dyedims_options = dyedims_options
            self.dyqbits_flag = dyqbits_flag
            self.dyqbits_options = dyqbits_options

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb)
                self.v_W_l = w_list
                if self.dyedims_flag or self.dyqbits_flag:
                    self.selector = EdimQbitSelectionNetwork(m_spa, 2*m_spa, len(self.dyedims_options), len(self.dyqbits_options))
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def create_mlp(self, ln, sigmoid_layer):
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            LL = nn.Linear(int(n), int(m), bias=True)

            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)

            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            n = ln[i]
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)
            
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def create_bare_emb(self, m, ln, weighted_pooling=None, dtype=torch.float32):
        emb_l = nn.ParameterList()
        for i in range(0, ln.size):
            n = ln[i]
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            E = nn.Parameter(torch.tensor(W).to(dtype), requires_grad=True)
            emb_l.append(E)
        return emb_l

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices, corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
                per_sample_weights=per_sample_weights,
            )

            ly.append(V)
        return ly

    def interact_features(self, x, ly):
        # Concatenate dense and sparse features
        (batch_size, d) = x.shape
        T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        # Perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        _, ni, nj = Z.shape
        offset = 1 if self.arch_interaction_itself else 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        Zflat = Z[:, li, lj]
        # Concatenate dense features and interactions
        R = torch.cat([x] + [Zflat], dim=1)
        return R

    def forward(self, dense_x, lS_o, lS_i):
        return self.sequential_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # batch size
        b = dense_x.size(0)

        # total sparse features
        n_spa = lS_i.size(0)

        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)

        # New: reduce dimensions of embeddings
        avg_dims = None
        if self.dyedims_flag:
            ly_s = torch.cat(ly, dim=0)
            reduced_dims, _ = self.selector(ly_s)

            values = reduced_dims @ torch.tensor(self.dyedims_options, device=reduced_dims.device).float()
            mask = torch.arange(max(self.dyedims_options), device=ly_s.device).unsqueeze(0) < values.unsqueeze(1)
            reduced_ly_s = ly_s * mask

            ly = list(reduced_ly_s.split(b, dim=0))
            avg_dims = values.sum() / b / n_spa

        # New: reduce quantization bits of embeddings
        avg_bits = None
        if self.dyqbits_flag:
            ly_s = torch.cat(ly, dim=0)
            _, reduced_bits = self.selector(ly_s)  # reduced_bits: [n_embeds, n_options], one-hot matrix
            # FP32
            ly_s_fp32 = ly_s * reduced_bits[:, 0].unsqueeze(1)
            # FP16
            ly_s_fp16 = FP16QuantDequantSTE(ly_s * reduced_bits[:, 1].unsqueeze(1))
            # FP8
            ly_s_fp8 = FP8QuantDequantSTE(ly_s * reduced_bits[:, 2].unsqueeze(1))
            quantized_ly_s = ly_s_fp32 + ly_s_fp16 + ly_s_fp8
            ly = list(quantized_ly_s.split(b, dim=0))
            avg_bits = (reduced_bits[:, 0].sum() * 32 + reduced_bits[:, 1].sum() * 16 + reduced_bits[:, 2].sum() * 8) / b / n_spa

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z, avg_dims, avg_bits


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )
    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )
    return value


def calculate_compression_rate(args, dlrm, log_iter=-1):
    #1. Get the selector network from DLRM
    selector = dlrm.selector

    #2. Fetch all 26 embedding tables
    emb_list = dlrm.emb_l

    #3. Feed all embeddings to selector and get back results
    dim_count = np.zeros(len(args.dyedims_options))
    bit_count = np.zeros(len(args.dyqbits_options))
    for emb_module in emb_list:
        embs = emb_module.weight.data
        dim_decs, bit_decs = selector(embs)
        dim_count += dim_decs.sum(0).detach().cpu().numpy()
        bit_count += bit_decs.sum(0).detach().cpu().numpy()

    #4. Calculate compression rate
    dim_compression_rate = np.array(args.dyedims_options).dot(dim_count) / (max(args.dyedims_options) * dim_count.sum())
    bit_compression_rate = np.array(args.dyqbits_options).dot(bit_count) / (max(args.dyqbits_options) * bit_count.sum())

    writer.add_scalar("Test/DimCompressionRate", dim_compression_rate, log_iter)
    writer.add_scalar("Test/BitCompressionRate", bit_compression_rate, log_iter)

    print(" dim compression rate {:3.3f} %, bit compression rate {:3.3f} %".format(
            dim_compression_rate * 100, bit_compression_rate * 100
        ),
        flush=True,
    )
    return dim_compression_rate, bit_compression_rate


def inference(
    args,
    dlrm,
    best_acc_test,
    test_ld,
    device,
    use_gpu,
    log_iter=-1,
):
    test_accu = 0
    test_samp = 0

    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break

        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )

        # forward pass
        Z_test, _, _ = dlrm_wrap(
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        ### gather the distributed results on each rank ###
        # For some reason it requires explicit sync before all_gather call if
        # tensor is on GPU memory
        if Z_test.is_cuda:
            torch.cuda.synchronize()

        # compute loss and accuracy
        S_test = Z_test.detach().cpu().numpy()  # numpy array
        T_test = T_test.detach().cpu().numpy()  # numpy array

        mbs_test = T_test.shape[0]  # = mini_batch_size except last
        A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

        test_accu += A_test
        test_samp += mbs_test

    acc_test = test_accu / test_samp
    writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    is_best = acc_test > best_acc_test
    if is_best:
        best_acc_test = acc_test
    print(
        " accuracy {:3.3f} %, best {:3.3f} %".format(
            acc_test * 100, best_acc_test * 100
        ),
        flush=True,
    )
    return model_metrics_dict, is_best


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation",
        type=str,
        choices=["random", "dataset", "internal"],
        default="random",
    )  # synthetic, dataset or internal
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")

    # Dynamic embedding dimensions
    parser.add_argument("--dyedims-flag", action="store_true", default=False)
    parser.add_argument("--dyedims-options", type=str, default="2,1")
    parser.add_argument("--dyedims-loss-temp", type=float, default=1.0)

    # Dynamic quantization bits
    parser.add_argument("--dyqbits-flag", action="store_true", default=False)
    parser.add_argument("--dyqbits-options", type=str, default="32,16,8")
    parser.add_argument("--dyqbits-loss-temp", type=float, default=1.0)

    global args
    global nbatches
    global nbatches_test
    global writer
    args = parser.parse_args()

    args.dyedims_options = list(map(int, args.dyedims_options.split(",")))
    args.dyqbits_options = list(map(int, args.dyqbits_options.split(",")))

    # Setup seeds
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        ngpus = torch.cuda.device_count()  # Limit the device count with `CUDA_VISIBLE_DEVICES`
        device = torch.device("cuda", 0)
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    # Prepare training data
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")

    if args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # Enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        else:
            ln_emb = np.array(ln_emb)
        m_den = train_data.m_den
        ln_bot[0] = m_den

    args.ln_emb = ln_emb.tolist()

    # Parse command line arguments
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_itself:
        num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
    else:
        num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out

    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # Sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if m_spa != m_den_out:
        sys.exit(
            "ERROR: arch-sparse-feature-size "
            + str(m_spa)
            + " does not match last dim of bottom mlp "
            + str(m_den_out)
        )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    global ndevices
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    # Construct the neural network specified above
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global dlrm
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        loss_function=args.loss_function,
        dyedims_flag=args.dyedims_flag,
        dyedims_options=args.dyedims_options,
        dyqbits_flag=args.dyqbits_flag,
        dyqbits_options=args.dyqbits_options,
    )

    if use_gpu:
        dlrm = dlrm.to(device)

    # Setup the optimizer
    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        opts = {
            "sgd": torch.optim.SGD,
            "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }
        optimizer = opts[args.optimizer](dlrm.parameters(), lr=args.learning_rate)

    # Training or inference
    best_acc_test = 0
    total_time = 0
    total_loss = 0
    total_dyedims_loss = 0
    total_dyqbits_loss = 0
    total_iter = 0
    total_samp = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        ld_model = torch.load(args.load_model, map_location=device) # device is also defined before based on the use_gpu arg
        dlrm.load_state_dict(ld_model["state_dict"], strict=False)
        # Freeze embedding weights
        for name, param in dlrm.named_parameters():
            if name.startswith("emb"):
                param.requires_grad = False

    print("Time/Loss/Accuracy (if enabled):")

    writer = SummaryWriter("./" + args.tensor_board_filename)

    if not args.inference_only:
        k = 0
        while k < args.nepochs:
            for j, inputBatch in enumerate(train_ld):

                X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

                t1 = time_wrap(use_gpu)

                # Early exit if nbatches was set by the user and has been exceeded
                if nbatches > 0 and j >= nbatches:
                    break

                mbs = T.shape[0]  # = args.mini_batch_size except maybe for last

                # Forward pass
                Z, avg_dims, avg_bits = dlrm_wrap(
                    X,
                    lS_o,
                    lS_i,
                    use_gpu,
                    device,
                    ndevices=ndevices,
                )

                # Loss
                E = loss_fn_wrap(Z, T, device)
                if args.dyedims_flag:
                    E += args.dyedims_loss_temp * avg_dims
                if args.dyqbits_flag:
                    E += args.dyqbits_loss_temp * avg_bits

                # Dump loss values
                L = E.detach().cpu().numpy()  # numpy array
                if args.dyedims_flag:
                    L1 = (args.dyedims_loss_temp * avg_dims).detach().cpu().numpy()
                if args.dyqbits_flag:
                    L2 = (args.dyqbits_loss_temp * avg_bits).detach().cpu().numpy()

                # Backward pass
                optimizer.zero_grad()
                E.backward()
                optimizer.step()

                t2 = time_wrap(use_gpu)
                total_time += t2 - t1

                # Dump total loss
                total_loss += L * mbs
                if args.dyedims_flag:
                    total_dyedims_loss += L1 * mbs
                if args.dyqbits_flag:
                    total_dyqbits_loss += L2 * mbs
                total_iter += 1
                total_samp += mbs

                # Print time, loss and accuracy
                should_print = ((j + 1) % args.print_freq == 0) or (
                    j + 1 == nbatches
                )
                should_test = (
                    (args.test_freq > 0)
                    and (args.data_generation in ["dataset", "random"])
                    and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )
                if should_print or should_test:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    train_loss = total_loss / total_samp
                    train_dyedims_loss = total_dyedims_loss / total_samp
                    train_dyqbits_loss = total_dyqbits_loss / total_samp
                    total_loss = 0
                    total_dyedims_loss = 0
                    total_dyqbits_loss = 0

                    str_run_type = (
                        "inference" if args.inference_only else "training"
                    )

                    wall_time = ""
                    if args.print_wall_time:
                        wall_time = " ({})".format(time.strftime("%H:%M"))

                    print(
                        "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                            str_run_type, j + 1, nbatches, k, gT
                        )
                        + " loss {:.6f},".format(train_loss)
                        + " dyedims loss {:.6f}".format(train_dyedims_loss)
                        + " dyqbits loss {:.6f}".format(train_dyqbits_loss)
                        + wall_time,
                        flush=True,
                    )

                    log_iter = nbatches * k + j + 1
                    writer.add_scalar("Train/Loss", train_loss, log_iter)
                    if args.dyedims_flag:
                        writer.add_scalar("Train/DyEDimsLoss", train_dyedims_loss, log_iter)
                    if args.dyqbits_flag:
                        writer.add_scalar("Train/DyQBitsLoss", train_dyqbits_loss, log_iter)

                    # Check gradient values in tensorboard
                    # for name, param in dlrm.selector.named_parameters():
                    #     if param.grad is not None:
                    #         writer.add_histogram(f"{name}.grad", param.grad, log_iter)

                    total_iter = 0
                    total_samp = 0

                # Test
                if should_test:
                    print("Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k))
                    model_metrics_dict, is_best = inference(
                        args,
                        dlrm,
                        best_acc_test,
                        test_ld,
                        device,
                        use_gpu,
                        log_iter,
                    )
                    if is_best:
                        best_acc_test = model_metrics_dict["test_acc"]

                    # Calculate the overall compression rate
                    if args.dyedims_flag or args.dyqbits_flag:
                        calculate_compression_rate(args, dlrm, log_iter)

                    if (is_best
                        and not (args.save_model == "")
                        and not args.inference_only
                    ):
                        model_metrics_dict["epoch"] = k
                        model_metrics_dict["iter"] = j + 1
                        model_metrics_dict["train_loss"] = train_loss
                        model_metrics_dict["total_loss"] = total_loss
                        model_metrics_dict["opt_state_dict"] = (
                            optimizer.state_dict()
                        )
                        print("Saving model to {}".format(args.save_model+".epoch{}.iter{}".format(k, j+1)))
                        torch.save(model_metrics_dict, args.save_model+".epoch{}.iter{}".format(k, j+1))
            k += 1
    else:
        print("Testing for inference only")
        inference(
            args,
            dlrm,
            best_acc_test,
            test_ld,
            device,
            use_gpu,
        )

if __name__ == "__main__":
    run()
