import numpy as np
import h5py
import time
import logging
#import matplotlib.pyplot as plt

from utilities import calculate_scalar, scale, inverse_scale


class DataGenerator(object):

    def __init__(self, hdf5_path, target_device, train_house_list,
        validate_house_list, batch_size, seq_len, width, random_seed=1234):
        """Data generator.

        Args:
          hdf5_path: string, path of hdf5 file.
          target_device: string, e.g. 'washingmachine'
          train_house_list: list of string, e.g. ['house1', 'house2']
          validate_house_list: list of string, e.g. ['house3']
          batch_size: int
          seq_len: int
          width: int
          random_seed: int
        """

        self.target_device = target_device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.width = width
        self.random_state = np.random.RandomState(random_seed)
        self.validate_house_list = validate_house_list

        assert len(train_house_list) > 0

        # Load hdf5 file
        load_time = time.time()

        self.hf = h5py.File(hdf5_path, 'r')

        # Load training data
        (self.train_x, self.train_y) = self.read_data(
            self.hf, target_device, train_house_list)

        # Load validation data
        (self.validate_x, self.validate_y) = self.read_data(
            self.hf, target_device, validate_house_list)

        logging.info("Load data time: {} s".format(time.time() - load_time))

        # Calculate scalar
        (self.mean, self.std) = calculate_scalar(self.train_x)

        # Training indexes
        self.train_indexes = np.arange(
            0, len(self.train_x) - seq_len - width + 2, 1)

        self.train_indexes = self.valid_data(self.train_x, self.train_y, self.train_indexes)


        # Validation indexes
        self.validate_indexes = np.arange(
            0, len(self.validate_x) - seq_len - width + 2, 1)

        self.validate_indexes = self.valid_data(self.validate_x, self.validate_y, self.validate_indexes)


        logging.info("Number of indexes: {}".format(len(self.train_indexes)))


    def valid_data(self, inputs, outputs, indexes):
        """remove invalid records: aggregate is 0 or less than the total of individual appliances
        """

        length = len(indexes)

        for i in range(length):

            if inputs[i] < outputs[i]:

                indexes[max(i - self.seq_len // 2, 0):min(i + self.seq_len // 2, length)] = -1

        valid_indexes = np.array([j for j in indexes if (j!=-1)])

        return valid_indexes



    def read_data(self, hf, target_device, house_list):
        """Load data of houses
        """

        if len(house_list) == 0:
            return [], []

        else:
            aggregates = []
            targets = []

            for house in house_list:

                aggregate = hf[house]['aggregate'][:]
                target = hf[house][target_device][:]

                aggregates.append(aggregate)
                targets.append(target)

            aggregates = np.concatenate(aggregates, axis=0)
            targets = np.concatenate(targets, axis=0)

            return aggregates, targets

    def generate(self):
        """Generate mini-batch data for training.
        """

        batch_size = self.batch_size

        indexes = np.array(self.train_indexes)
        self.random_state.shuffle(indexes)

        iteration = 0
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= len(indexes):
                pointer = 0

                self.random_state.shuffle(indexes)

            # Get batch indexes
            batch_indexes = indexes[pointer : pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len + self.width - 1)
            batch_y_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len // 2, self.seq_len // 2 + self.width)

            batch_x = self.train_x[batch_x_indexes_2d]
            batch_y = self.train_y[batch_y_indexes_2d]

            # Transform
            batch_x = self.transform(batch_x)
            batch_y = self.transform(batch_y)

            yield batch_x, batch_y

    def generate_validate(self, data_type, max_iteration, shuffle=True):
        """Generate mini-batch data for validation.
        """

        batch_size = self.batch_size

        if data_type == 'train':
            indexes = np.array(self.train_indexes)
            x = self.train_x
            y = self.train_y

        elif data_type == 'validate':
            assert len(self.validate_house_list) > 0
            indexes = np.array(self.validate_indexes)
            x = self.validate_x
            y = self.validate_y

        else:
            raise Exception("Incorrect data_type!")

        if shuffle:
            self.random_state.shuffle(indexes)

        iteration = 0
        pointer = 0

        while pointer < len(indexes):

            # Reset pointer
            if iteration == max_iteration:
                break

            # Get batch indexes
            batch_indexes = indexes[pointer : pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len + self.width - 1)
            batch_y_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len // 2, self.seq_len // 2 + self.width)

            batch_x = x[batch_x_indexes_2d]
            batch_y = y[batch_y_indexes_2d]

            # Transform
            batch_x = self.transform(batch_x)
            batch_y = self.transform(batch_y)

            yield batch_x, batch_y


    def transform(self, x):
        return scale(x, self.mean, self.std)

    def inverse_transform(self, x):
        return inverse_scale(x, self.mean, self.std)



class TestDataGenerator(DataGenerator):

    def __init__(self, hdf5_path, target_device, train_house_list, seq_len, steps):
        """Test data generator.

        Args:
          hdf5_path: string, path of hdf5 file.
          target_device: string, e.g. 'washingmachine'
          train_house_list: list of string, e.g. ['house1', 'house2']
          seq_len: int
          steps: int
        """

        DataGenerator.__init__(self,
                               hdf5_path=hdf5_path,
                               target_device=target_device,
                               train_house_list=train_house_list,
                               validate_house_list=[],  # dummy arg
                               batch_size=None,     # dummy arg
                               seq_len=seq_len,
                               width=1,     # dummy arg
                               random_seed=None, # dummy arg
                               )

        self.steps = steps

    def generate_inference(self, house):
        """Generate data for inference.
        """

        seq_len = self.seq_len
        steps = self.steps

        x, y = self.read_data(self.hf, self.target_device, house_list=[house])
        self.target = y
        self.source = x

        x = self.pad_seq(x)

        index = 0

        while (index + seq_len <= len(x)):
            batch_x = x[index : index + seq_len + steps - 1]
            batch_x = batch_x[np.newaxis, :]

            batch_x = self.transform(batch_x)

            index += steps
            yield batch_x

    def pad_seq(self, x):
        """Pad the boundary of a sequence.
        """
        pad_num = self.seq_len // 2
        return np.concatenate((np.zeros(pad_num), x, np.zeros(pad_num)))

    def get_target(self):
        return self.target

    def get_source(self):
        return self.source

    def inverse_transform(self, x):
        return inverse_scale(x, self.mean, self.std)
