from models.adaptive_fast_xgboost_multiclass import AdaptiveMulticlass

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import RandomRBFGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.data import LEDGenerator
from skmultiflow.data.file_stream import FileStream

# parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.3     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 10000 # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
small_window_size = 150
detect_drift = True     # Enable/disable drift detection
max_buffer = 25
pre_train = 15
use_updater = True
num_class = 7

# classifier
AFXGBMC = AdaptiveMulticlass(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  small_window_size=small_window_size,
                                  max_buffer=max_buffer,
                                  pre_train=pre_train,
                                  detect_drift=detect_drift,
                                  use_updater=use_updater,
                                  num_class=num_class)

# dataset
# stream = RandomRBFGenerator(n_classes=4,model_random_state=1,sample_random_state=1)
stream = SEAGenerator(noise_percentage=0.1)
# stream = LEDGenerator(random_state=1, noise_percentage=0.28, has_noise=True)
# stream = FileStream("./datasets/.csv")

evaluator = EvaluatePrequential(pretrain_size=0,
                                max_samples=200000,
                                # batch_size=1,
                                show_plot=False,
                                metrics=["accuracy","running_time"])

evaluator.evaluate(stream=stream,
                  model=[AFXGBMC],
                  model_names=["AFXGB-MC"])
