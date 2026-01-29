# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# will be releasted soon

from semilearn.algorithms import name2alg
from semilearn.core.utils import IMB_ALGORITHMS

# 导入所有算法以触发注册
from . import supervise, long_tail_la, fixmatch, freematch, softmatch, acr, simpro, cdmad, cpg, ours

name2imbalg = IMB_ALGORITHMS


def get_imb_algorithm(args, net_builder, tb_log, logger):
    if args.imb_algorithm not in name2imbalg:
        print(f'Unknown imbalanced algorithm: {args.imb_algorithm }')

    class DummyClass(name2imbalg[args.imb_algorithm], name2alg[args.algorithm]):
        def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
            # 先调用不平衡算法的初始化
            name2imbalg[args.imb_algorithm].__init__(self, args, net_builder, tb_log, logger, **kwargs)
    
    alg = DummyClass(args=args, net_builder=net_builder, tb_log=tb_log, logger=logger)
    return alg