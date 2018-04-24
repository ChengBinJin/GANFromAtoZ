class Day2Night(object):
    def __init__(self, flags):
        self.flags = flags
        self.name = 'day2night'
        self.image_size = (256, 512, 3)

        self.day_path = '../data/ICVL/day/'
        self.night_path = '../data/ICVL/night/'

        # tfrecord path
        self.day_tfpath = '../data/tfrecords/day.tfrecords'
        self.night_tfpath = '../data/tfrecords/night.tfrecords'

    def __call__(self):
        return [self.day_tfpath, self.night_tfpath]


# noinspection PyPep8Naming
def Dataset(dataset_name, flags):
    if dataset_name == 'day2night':
        return Day2Night(flags)
    else:
        raise NotImplementedError
