import numpy as np
import h5py
import time
import logging
#import matplotlib.pyplot as plt

from utilities import calculate_scalar, scale, inverse_scale, binarize

#Data generator for seq_to_seq_lstm_model

class DataGenerator(object):

    def __init__(self, hdf5_path, target_device, train_house_list, validate_house_list,
                 batch_size, seq_len, width, random_seed=1234, binary_threshold=None,
                 balance_threshold=None, balance_positive=0.5):
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
          binary_threshold: None or float
        """

        self.target_device = target_device
        self.batch_size = batch_size
        assert seq_len % 2 == 1, "seq_len has to be odd, otherwise padding will be off by 1"
        self.seq_len = seq_len
        self.width = width
        self.random_state = np.random.RandomState(random_seed)
        self.validate_random_state = np.random.RandomState(1)
        self.validate_house_list = validate_house_list
        self.binary_threshold = binary_threshold
        self.balance_threshold = balance_threshold
        self.balance_positive = balance_positive

        if self.balance_threshold is not None:
            logging.info('----balance threshold is not none and use balance generation----')
            self.generate = self._generate_balanced
        else:
            logging.info('----balance threshold is none and normal generation is used----')
            self.generate = self._generate

        assert len(train_house_list) > 0

        # Load hdf5 file
        load_time = time.time()

        self.hf = h5py.File(hdf5_path, 'r')

        # Load training data
        # NOTE read_data will return binarized data if binary_threshold is set to a float
        (self.train_x, self.train_y) = self.read_data(
            self.hf, target_device, train_house_list)

        # Load validation data
        # NOTE read_data will return binarized data if binary_threshold is set to a float
        (self.validate_x, self.validate_y) = self.read_data(
            self.hf, target_device, validate_house_list)

        logging.info("Load data time: {} s".format(time.time() - load_time))

        # Calculate scalar
        (self.mean_x, self.std_x, self.max_x) = calculate_scalar(self.train_x)
        (self.mean_y, self.std_y, self.max_y) = calculate_scalar(self.train_y)

        logging.info('mean_x: {}, std_x: {}, max_x: {}'.format(self.mean_x, self.std_x, self.max_x))
        logging.info('mean_y: {}, std_y: {}, max_y: {}:'.format(self.mean_y, self.std_y, self.max_y))

        # Training indexes
        self.train_indexes = np.arange(
            0, len(self.train_x) - seq_len - width + 2, width)

        self.train_indexes = self.valid_data(self.train_x, self.train_y, self.train_indexes)


        # Validation indexes
        self.validate_indexes = np.arange(
            0, len(self.validate_x) - seq_len - width + 2, width)

        self.validate_indexes = self.valid_data(self.validate_x, self.validate_y, self.validate_indexes)


        logging.info("Number of indexes: {}".format(len(self.train_indexes)))


    def valid_data(self, inputs, outputs, indexes):
        """remove invalid records: aggregate is less than the consumption of individual appliances
           we do not remove the data where the aggregate is 0 as this will remove too much data for training (more than 90%).
        """

        full_indexes = np.arange(0, len(inputs) - self.seq_len - self.width + 2, 1)
        length = len(full_indexes)

        no_of_invalid = 0
        zero_output = 0
        for i in range(length):

            if inputs[i] < outputs[i]:
                no_of_invalid = no_of_invalid + 1
                full_indexes[max(i - self.seq_len // 2, 0):min(i + self.seq_len // 2, length)] = -1
            if outputs[i]>4 and outputs[i]<5:
                zero_output = zero_output + 1

        valid_indexes = np.array([j for j in indexes if (full_indexes[j]!=-1)])
        logging.info('number of full indexes: {}'.format(len(indexes)))
        logging.info('propotion of valid indexes: {}'.format(1.0*len(valid_indexes)/(np.finfo(float).eps+len(indexes))))
        logging.info('number of valid indexes: {}'.format(len(valid_indexes)))
        logging.info('propotion of zero outputs: {}'.format(1.0*zero_output/(np.finfo(float).eps+len(outputs))))
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

    def _generate_balanced(self):
        """Generate mini-batch data for training using balanced data.
        """
        logging.info('----balance generation----')
        batch_size = self.batch_size

        indexes = np.array(self.train_indexes)

        positive_size = int(self.batch_size * self.balance_positive)

        target_values = self.train_y[indexes]
        indexes_on = indexes[target_values >= self.balance_threshold]
        indexes_off = indexes[target_values < self.balance_threshold]

        i_on = len(indexes_on)  # To trigger shuffling
        i_off = len(indexes_off)  # To trigger shuffling
        while True:

            if i_on + positive_size > len(indexes_on):
                i_on = 0
                self.random_state.shuffle(indexes_on)

            if i_off + batch_size - positive_size > len(indexes_off):
                i_off = 0
                self.random_state.shuffle(indexes_off)

            # Get batch indexes
            batch_indexes = np.concatenate((indexes_on[i_on:i_on+positive_size], indexes_off[i_off:i_off+batch_size-positive_size]), axis=0)
            batch_x_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len)
            batch_y_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len)
            batch_x = self.train_x[batch_x_indexes_2d]
            batch_y = self.train_y[batch_y_indexes_2d]
            # Normalize input
            batch_x = self.transform(batch_x)
            if self.binary_threshold is not None:
                batch_y = binarize(batch_y, self.binary_threshold)
            else:
                batch_y = self.transform(batch_y)

            yield batch_x, batch_y
            i_on += positive_size
            i_off += batch_size-positive_size


    def _generate(self):
        """Generate mini-batch data for training.
        """
        logging.info('----no balance generation----')

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

            batch_x_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len)
            batch_y_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len)

            batch_x = self.train_x[batch_x_indexes_2d]
            batch_y = self.train_y[batch_y_indexes_2d]

            # Normalize input
            batch_x = self.transform(batch_x)
            if self.binary_threshold is not None:
                batch_y = binarize(batch_y, self.binary_threshold)
            else:
                batch_y = self.transform(batch_y)
            #logging.info("batch_x batch_y shape: ", batch_x.shape, batch_y.shape)

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
            self.validate_random_state.shuffle(indexes)

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

            batch_x_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len)
            batch_y_indexes_2d = batch_indexes[:, None] + np.arange(self.seq_len)

            batch_x = x[batch_x_indexes_2d]
            batch_y = y[batch_y_indexes_2d]

            # Normalize input
            batch_x = self.transform(batch_x)
            if self.binary_threshold is not None:
                batch_y = binarize(batch_y, self.binary_threshold)
            else:
                batch_y = self.transform(batch_y)

            yield batch_x, batch_y


    def transform(self, x):
        return scale(x, self.mean_x, self.std_x)

    def inverse_transform(self, x):
        return inverse_scale(x, self.mean_x, self.std_x)



class TestDataGenerator(DataGenerator):

    def __init__(self, hdf5_path, target_device, train_house_list, seq_len, steps, binary_threshold=None):
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
                               binary_threshold=binary_threshold
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

        #x = self.pad_seq(x)

        index = 0

        while (index + seq_len <= len(x)):
            batch_x = x[index : index + seq_len + steps*200 - 1]
            batch_x = batch_x[np.newaxis, :]

            batch_x = self.transform(batch_x)

            index = index + seq_len + steps*200 - 1
            yield batch_x

    def pad_seq(self, x):
        """Pad the boundary of a sequence.
        """
        pad_num = self.seq_len // 2
        return np.concatenate((np.zeros(pad_num), x, np.zeros(pad_num)))

    def get_target(self):
        if self.binary_threshold is not None:
            return binarize(self.target, self.binary_threshold)
        return self.target

    def get_source(self):
        return self.source

    def inverse_transform(self, x):
        return inverse_scale(x, self.mean_x, self.std_x)
