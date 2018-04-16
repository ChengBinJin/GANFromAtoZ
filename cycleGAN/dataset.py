class Day2Night(object):
    def __init__(self, flags):
        self.flags = flags
        self.image_size = (128, 256, 3)

        self.day_path = '../data/tfrecords/day.tfrecords'
        self.night_path = '../data/tfrecords/night.tfrecords'

    def __call__(self):
        return [self.day_path, self.night_path]


# noinspection PyPep8Naming
def Dataset(dataset_name, flags):
    if dataset_name == 'day2night':
        return Day2Night(flags)
    else:
        raise NotImplementedError
