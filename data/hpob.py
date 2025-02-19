import os
import json
import torch
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
import botorch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch import ExactMarginalLogLikelihood


class HPOBHandler:
    def __init__(
        self,
        root_dir: str,
        mode="v3-train-augmented",
    ):
        """Data handler for HPO-B tasks.

        Args:
            root_dir, str: where HPO-B benchmark data files are stored.
            mode, str: HPO-B benchmark provides 5 different modes: v1, v2, v3, v3-test, and v3-train-augmented.

        Attrs:
            seeds, list: 5 seeds for initialization during test time.
        """
        print("Loading HPO-B handler...")
        self.mode = mode
        self.seeds = ["test0", "test1", "test2", "test3", "test4"]
        self.root_dir = root_dir

        # only v3-test and v3-train-augmented are used in our project
        if self.mode == "v3-test":
            self.load_data(root_dir, only_test=True)
        elif self.mode == "v3-train-augmented":
            self.load_data(root_dir, only_test=False, augmented_train=True)
        elif self.mode in ["v1", "v2", "v3"]:
            raise NotImplementedError
        else:
            raise ValueError("Provide a valid mode.")

    def load_data(
        self,
        rootdir: str = "",
        only_test: bool = True,
        augmented_train: bool = False,
    ):
        """Load data from HPO-B benchmark.

        Args:
            root_dir, str: where HPO-B benchmark data files are stored.
            only_test, bool: whether to load only test data.
            augmented_train, bool: whether to load the augmented train data. dummy.

        Attrs:
            meta_train_data:
            meta_train_data_pred_idx:
            meta_train_data_acq_idx:
            meta_test_data:
            meta_validation_data:
            search_space_dims:
        """

        print("Loading data...")
        meta_train_augmented_path = os.path.join(
            rootdir, "meta-train-dataset-augmented.json"
        )
        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir, "meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")

        # load test data
        with open(meta_test_path, "rb") as f:
            self.meta_test_data = json.load(f)

        # load training data
        if not only_test:
            if augmented_train:
                with open(meta_train_augmented_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            else:
                with open(meta_train_path, "rb") as f:
                    self.meta_train_data = json.load(f)

            # split each dataset in all search spaces into two not overlapped parts for prediction and acquisition tasks
            (
                self.meta_train_data_pred_idx,
                self.meta_train_data_acq_idx,
            ) = self.random_split(data=self.meta_train_data)

            # load validation data
            with open(meta_validation_path, "rb") as f:
                self.meta_validation_data = json.load(f)

        self.search_space_dims = {}

        # load information regarding search space dimensions
        for search_space in self.meta_test_data.keys():
            dataset = list(self.meta_test_data[search_space].keys())[0]
            X = self.meta_test_data[search_space][dataset]["X"][0]
            self.search_space_dims[search_space] = len(X)

    def random_split(self, data: Dict, split: float = 0.5) -> Tuple[Dict, Dict]:
        """Randomly split `data` into two parts according to the ratio given by `split`.

        Args:
            data, Dict{"search_space_id": {"dataset_id": {"X": List, "y": List}}}.
            ratio, float: the data partition.

        Returns:
            pred_idx, Dict{"search_space_id": {"dataset_id": List}}
            acq_idx, Dict{"search_space_id": {"dataset_id": List}}

        """
        print("Split each dataset into prediction and acquistion parts...")
        pred_idx, acq_idx = {}, {}
        # enumerate all search spaces
        for search_space in data.keys():
            pred_idx[search_space], acq_idx[search_space] = (
                {},
                {},
            )

            # enumerate all datasets on the search space
            for dataset in data[search_space].keys():
                # the number of samples in this dataset
                total = len(data[search_space][dataset]["X"])
                # random permutation
                perm_idx = np.random.permutation(total)

                pred_idx[search_space][dataset] = perm_idx[: int(total * split)]
                acq_idx[search_space][dataset] = perm_idx[int(total * split) :]

        return pred_idx, acq_idx

    def get_gp_on_task(
        self, split: str, search_space_id: int, task_id: int
    ) -> Tuple[SingleTaskGP, int]:
        """Fit a GP on specified dataset on the specified search space.
        Reference: https://github.com/huawei-noah/HEBO/tree/master/NAP

        Returns:
            model, SingleTaskGP: the fitted GP.
            num_benchmark_points, int: the number of benchmark samples.
        """
        # find benchmark data points
        split_dict = {
            "train": self.meta_train_data,
            "val": self.meta_validation_data,
            "test": self.meta_test_data,
        }
        X = np.array(split_dict[split][search_space_id][task_id]["X"])
        y = np.array(split_dict[split][search_space_id][task_id]["y"])
        std_y = MinMaxScaler().fit_transform(y)

        # we save the GP
        gp_name = f"{self.mode}_{search_space_id}_{task_id}_gp.pt"
        gp_model_path = os.path.join(self.root_dir, "gps", gp_name)
        if not os.path.exists(gp_model_path):
            print(
                f"fit GP on dataset with dataset id {task_id} on space {search_space_id}, containing {X.shape[0]} points..."
            )
            normX = torch.from_numpy(X).to(dtype=torch.float64)
            std_y = torch.from_numpy(std_y).to(dtype=torch.float64)

            # fit GP
            model = SingleTaskGP(train_X=normX, train_Y=std_y.view(-1, 1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                mll.cpu()
                _ = fit_gpytorch_mll(mll=mll)
            except (RuntimeError, botorch.exceptions.errors.ModelFittingError) as e:
                print(e)
                try:
                    print("Try fit on GPU.")
                    mll.cuda()
                    _ = fit_gpytorch_mll_torch(mll)
                except RuntimeError as e:
                    print(
                        f"Error during the GP fit on search space {search_space_id}, dataset {task_id}."
                    )
                    print(e)

                    # free GPU
                    normX = normX.cpu().numpy()
                    std_y = std_y.cpu().numpy()
                    model = model.cpu()
                    mll = mll.cpu()
                    del model, mll
                    torch.cuda.empty_cache()

            # save GPs
            with torch.no_grad():
                if not os.path.exists(os.path.join(self.root_dir, "gps")):
                    os.makedirs(os.path.join(self.root_dir, "gps"), exist_ok=True)
                torch.save(model, os.path.join(self.root_dir, "gps", gp_name))
                print(f"saved model at {gp_model_path}")

                # free GPU
                normX = normX.cpu()
                std_y = std_y.cpu()
                mll = mll.cpu()
                model.eval()
                del normX, std_y, mll  # NOTE don't delete the model...
                torch.cuda.empty_cache()

        else:
            model = torch.load(gp_model_path)
            model.eval()

        return model, X.shape[1]

    def sample_from_gp_posterior(
        self, split: str, search_space_id: int, task_id: int, num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and sample from GP posterior on benchmark data.

        Returns:
            test_X, (num_points, d_x): sampled inputs.
            test_y, (num_points, 1): correspoding function values sampled from fitted GP.
        """
        # get GP
        model, xdim = self.get_gp_on_task(split, search_space_id, task_id)

        # draw samples from the GP
        test_X = torch.rand(num_points, xdim)
        model = model.to(test_X)
        test_y = (
            model.posterior(test_X).sample().view(num_points, 1)
        )  # (num_points, d_x)

        # free GPU
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
        return test_X.cpu().numpy(), test_y.cpu().numpy()

    def sample_a_task(
        self,
        search_space_id: int,
        task_id: int,
        num_points: int,
        split: str,
        train_split: str,
        standardize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """sample samples from a specified dataset on a specified search space.

        Args:
            search_space_id, int: ID of search space.
            batch_size, int: number of datasets to sample in specified search space. All datasets are returned if `batch_size` is larger than the total number.
            num_points, int:  number of samples in each dataset. The entire dataset is returned if `num_points` is larger than dataset size.
            standardize, bool: whether to standardize y.
            split, str in ["train", "val", "test"]: sample training, validation, or test data.
            train_split, str in ["pred", "acq"]: sample training data for prediction or acquisition tasks.

        Returns:
            X, (num_points, d_x): inputs for `num_points` samples.
            y, (num_points, 1): corresponding function values.
            Xopt, (1, d_x): input corresponding to the maximal function value.
            yopt, (1, 1), : maximal function value from this sampled subset.
        """
        try:
            if split == "train":
                data = self.meta_train_data
            elif split == "val":
                data = self.meta_validation_data
            else:
                data = self.meta_test_data

            X = np.array(data[search_space_id][task_id]["X"])
            y = np.array(data[search_space_id][task_id]["y"])
        except KeyError:
            print(data.keys())
            raise

        # during training, get prediction samples or acquisition samples from predefined indices
        if split == "train":
            idx = (
                self.meta_train_data_pred_idx[search_space_id][task_id]
                if train_split == "pred"
                else self.meta_train_data_acq_idx[search_space_id][task_id]
            )
            X, y = X[idx], y[idx]

        # fit and sample from GPs when the number of points are not fewer than required
        if split == "train" and len(X) < num_points:
            # TODO no check for val and test data
            X, y = self.sample_from_gp_posterior(
                split=split,
                search_space_id=search_space_id,
                task_id=task_id,
                num_points=num_points,
            )

        # generate `num_points` samples
        perm_idx = np.random.permutation(len(X))[:num_points]
        X, y = X[perm_idx], y[perm_idx]

        # standardize the y
        if standardize:
            y = MinMaxScaler().fit_transform(y)

        # find maximum
        yopt = np.max(y, axis=0, keepdims=True)  # (1, 1)
        Xopt = np.take_along_axis(X, np.argmax(y, axis=0, keepdims=True), axis=0)

        return (X, y, Xopt, yopt)

    def sample(
        self,
        search_space_id: int,
        num_points: int,
        batch_size: int = 1,
        standardize: bool = True,
        train_split: str = "acq",
        split: str = "train",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """sample a batch of data from specified search space.

        Args:
            search_space_id, int: ID of search space.
            batch_size, int: number of datasets to sample in specified search space. All datasets are returned if `batch_size` is larger than the total number.
            num_points, int:  number of samples in each dataset. The entire dataset is returned if `num_points` is larger than dataset size.
            standardize, bool: whether to standardize y.
            split, str in ["train", "val", "test"]: sample training, validation, or test data.
            train_split, str in ["pred", "acq"]: sample training data for prediction or acquisition tasks.

        Returns:
            X, (batch_size, num_points, d_x):
            y, (batch_size, num_points, 1):
            Xopt, (batch_size, 1, d_x): input corresponding to the maximal function value.
            yopt, (batch_size, 1, 1), : maximal function value.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Provide a valid data split.")
        if train_split not in ["pred", "acq"]:
            raise ValueError("Provide a valid training data split.")

        (
            _X,
            _Y,
            _XOPT,
            _YOPT,
        ) = ([], [], [], [])

        # sample task ids from the specified search space.
        # randomly sample `batch_size` datasets with replacement
        if split == "train":
            task_ids = np.random.choice(
                list(self.meta_train_data[search_space_id].keys()), batch_size
            )
        elif split == "val":
            task_ids = np.random.choice(
                list(self.meta_validation_data[search_space_id].keys()), batch_size
            )
        else:
            # evaluation on all datasets when testing
            print("All datasets are tested.")
            task_ids = list(self.meta_test_data[search_space_id].keys())

        for task_id in task_ids:
            # sample data from the task
            X, y, Xopt, yopt = self.sample_a_task(
                search_space_id=search_space_id,
                task_id=task_id,
                num_points=num_points,
                split=split,
                train_split=train_split,
                standardize=standardize,
            )

            _X.append(X)
            _Y.append(y)
            _XOPT.append(Xopt)
            _YOPT.append(yopt)

        return (
            np.stack(_X, axis=0),
            np.stack(_Y, axis=0),
            np.stack(_XOPT, axis=0),
            np.stack(_YOPT, axis=0),
        )
