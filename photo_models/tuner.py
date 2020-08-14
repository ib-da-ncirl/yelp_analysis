#  The MIT License (MIT)
#  Copyright (c) 2020. Ian Buttimer
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
import kerastuner as kt

from photo_models.model_misc import calc_step_size


class RandomSearchTuner(kt.RandomSearch):
    """Random search tuner.

    # Arguments:
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        model_args: ModelArgs. Model arguments.
        seed: Int. Random seed.
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
        tune_new_entries: Whether hyperparameter entries
            that are requested by the hypermodel
            but that were not specified in `hyperparameters`
            should be added to the search space, or not.
            If not, then the default value for these parameters
            will be used.
        allow_new_entries: Whether the hypermodel is allowed
            to request hyperparameter entries not listed in
            `hyperparameters`.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(self, hypermodel, objective, max_trials, model_args, **kwargs):

        super(RandomSearchTuner, self).__init__(hypermodel, objective, max_trials, **kwargs)
        self.model_args = model_args

    def run_trial(self, trial, *fit_args, **fit_kwargs):

        stb = self.model_args.split_total_batch()
        step_size_train = calc_step_size(self.model_args.train_data, stb, 'training')
        step_size_valid = calc_step_size(self.model_args.val_data, stb, 'validation')

        fit_args = fit_args + (self.model_args.train_data,)

        fit_kwargs['batch_size'] = self.model_args.batch_size
        fit_kwargs['steps_per_epoch'] = step_size_train
        fit_kwargs['epochs'] = self.model_args.epochs
        fit_kwargs['validation_data'] = self.model_args.val_data
        fit_kwargs['validation_steps'] = step_size_valid

        super(RandomSearchTuner, self).run_trial(trial, *fit_args, **fit_kwargs)

