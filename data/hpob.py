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
        Adapted from NAP (GNU APGL-3.0 License)
        Source: https://github.com/huawei-noah/HEBO/blob/master/NAP/HPOB_data/create_task_datasets.py

        Args:
            root_dir, str: where HPO-B benchmark data files are stored.
            mode, str: HPO-B benchmark provides 5 different modes: v1, v2, v3, v3-test, and v3-train-augmented.
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
            augmented_train, bool: whether to load the augmented train data.

        Attrs:
            meta_train_data, dict: training dataset.
            meta_train_data_pred_idx, dict: indices for the prediction task (split from `meta_train_data`).
            meta_train_data_acq_idx, dict: indices for the acquisition task (split from `meta_train_data`).
            meta_test_data, dict: test dataset.
            meta_validation_data, dict: validation dataset.
            search_space_dims, dict: dictionary mapping search spaces to their respective dimensionalities.
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
        """Randomly split `data` into two parts based on the given `split` ratio.

        Args:
            data, Dict: dictionary structured as
                {"search_space_id": {"dataset_id": {"X": List, "y": List}}}.
            split, float: proportion of the data assigned to the first partition (default: 0.5).

        Returns:
            pred_idx, Dict: dictionary mapping search space and dataset IDs to the first partition indices.
            acq_idx, Dict: dictionary mapping search space and dataset IDs to the second partition indices.
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
        """Fit a GP on a dataset."""
        split_dict = {
            "train": self.meta_train_data,
            "val": self.meta_validation_data,
            "test": self.meta_test_data,
        }
        X = np.array(split_dict[split][search_space_id][task_id]["X"])
        y = np.array(split_dict[split][search_space_id][task_id]["y"])
        std_y = MinMaxScaler().fit_transform(y)

        # save GP
        gp_name = f"{self.mode}_{search_space_id}_{task_id}_gp.pt"
        gp_model_path = os.path.join(self.root_dir, "gps", gp_name)
        if not os.path.exists(gp_model_path):
            print(
                f"fit GP on dataset {task_id} on space {search_space_id} with {X.shape[0]} points..."
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
                    mll.cuda()
                    _ = fit_gpytorch_mll_torch(mll)
                except RuntimeError as e:
                    print(
                        f"Error during fitting GP on search space {search_space_id}, dataset {task_id}."
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
                normX = normX.cpu().numpy()
                std_y = std_y.cpu().numpy()
                mll = mll.cpu()
                model.eval()
                del mll  # NOTE don't delete the model...
                torch.cuda.empty_cache()

        else:
            model = torch.load(gp_model_path, weights_only=False)
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
        """sample datapoints from a task (dataset) on a search space.

        Args:
            search_space_id, int: ID of search space.
            batch_size, int: number of tasks to sample. All datasets are returned if `batch_size` is larger than the total number.
            num_points, int:  number of datapoints in each dataset. The entire dataset is returned if `num_points` is larger than dataset size.
            standardize, bool: whether to standardize y.
            split, str: indicates the split ["train", "val", "test"] to sample from.
            train_split, str: if sample from trainning set, indicates the split in ["pred", "acq"] to sample from.

        Returns:
            X, (num_points, d_x): datapoints.
            y, (num_points, 1): utility values.
            Xopt, (1, d_x): global optimum locations.
            yopt, (1, 1), : global optimum values.
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
        search_space_id: str,
        num_total_points: int,
        batch_size: int = 1,
        standardize: bool = True,
        train_split: str = "acq",
        split: str = "train",
        device: str = "cuda",
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """sample a batch of data from specified search space.

        Args:
            search_space_id, int: ID of search space.
            batch_size, int: number of tasks to sample. All datasets are returned if `batch_size` is larger than the total number.
            num_points, int:  number of datapoints in each dataset. The entire dataset is returned if `num_points` is larger than dataset size.
            standardize, bool: whether to standardize y.
            split, str: indicates the split ["train", "val", "test"] to sample from.
            train_split, str: if sample from trainning set, indicates the split in ["pred", "acq"] to sample from.
            device, str: computational device in ["cpu", "cuda"].

        Returns:
            X, (batch_size, num_points, d_x): datapoints.
            y, (batch_size, num_points, 1): utility values.
            Xopt, (batch_size, 1, d_x): global optimum locations.
            yopt, (batch_size, 1, 1), : global optimum values.
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
                num_points=num_total_points,
                split=split,
                train_split=train_split,
                standardize=standardize,
            )

            _X.append(X)
            _Y.append(y)
            _XOPT.append(Xopt)
            _YOPT.append(yopt)

        return (
            torch.from_numpy(np.stack(_X, axis=0)).float().to(device),
            torch.from_numpy(np.stack(_Y, axis=0)).float().to(device),
            torch.from_numpy(np.stack(_XOPT, axis=0)).float().to(device),
            torch.from_numpy(np.stack(_YOPT, axis=0)).float().to(device),
        )
