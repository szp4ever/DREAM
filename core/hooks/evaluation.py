# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("\n" + "=" * 80)
            algorithm.print_fn(f"Epoch {algorithm.epoch + 1} - Validation Results:")
            algorithm.print_fn("=" * 80)
            
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            # 输出详细验证结果
            algorithm.print_fn(f"  Top-1 Accuracy: {eval_dict['eval/top-1-acc']:.4f}")
            algorithm.print_fn(f"  Top-5 Accuracy: {eval_dict['eval/top-5-acc']:.4f}")
            algorithm.print_fn(f"  Balanced Accuracy: {eval_dict['eval/balanced_acc']:.4f}")
            algorithm.print_fn(f"  Macro Precision: {eval_dict['eval/precision']:.4f}")
            algorithm.print_fn(f"  Macro Recall: {eval_dict['eval/recall']:.4f}")
            algorithm.print_fn(f"  Macro F1: {eval_dict['eval/F1']:.4f}")
            algorithm.print_fn(f"  Loss: {eval_dict['eval/loss']:.4f}")

            # update best metrics
            if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc']
                algorithm.best_it = algorithm.it
                algorithm.print_fn(f"  🎉 New best accuracy: {algorithm.best_eval_acc:.4f} at iteration {algorithm.best_it}")
            else:
                algorithm.print_fn(f"  Best accuracy so far: {algorithm.best_eval_acc:.4f} (at iteration {algorithm.best_it})")
            
            algorithm.print_fn("=" * 80)
            algorithm.print_fn("")

    def after_run(self, algorithm):
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict
