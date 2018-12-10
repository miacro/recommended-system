import os
import glob
import tensorflow as tf


class Dataset():
    def __init__(self, directory):
        self.directory = directory
        self.movieids = {}
        self.consumerids = {}
        self.probe_set = {}

    def read_file(self, filename):
        with open(filename, "rt") as file:
            leader = ""
            for line in file.readlines():
                line = line.strip()
                if line[-1] == ":":
                    leader = line[:-1]
                else:
                    contents = line.split(",")
                    yield [leader] + contents

    def load_training_set(self):
        probefile = os.path.join(self.directory, "probe.txt")
        for movieid, consumerid in self.read_file(probefile):
            if movieid not in self.probe_set:
                self.probe_set[movieid] = {}
            self.probe_set[movieid][consumerid] = {"rate": 0, "date": None}

        directory = os.path.join(self.directory, "training_set")
        filenames = glob.glob(os.path.join(directory, "*.txt"))
        filenames.sort()
        for filename in filenames:
            for movieid, consumerid, rate, date in self.read_file(filename):
                if movieid not in self.movieids:
                    self.movieids[movieid] = len(self.movieids)
                if consumerid not in self.consumerids:
                    self.consumerids[consumerid] = len(self.consumerids)
                movieindex = self.movieids[movieid]
                consumerindex = self.consumerids[consumerid]

                if movieid in self.probe_set:
                    if consumerid in self.probe_set[movieid]:
                        self.probe_set[movieid][consumerid] = {
                            "rate": float(rate),
                            "date": date
                        }

                yield movieindex, consumerindex, float(rate), date
            print("file: {} done".format(filename))

    def load_qualifying_set(self):
        filename = os.path.join(self.directory, "qualifying.txt")
        for movieid, consumerid, date in self.read_file(filename):
            yield self.movieids[movieid], self.consumerids[consumerid], date

    def load_probe_set(self):
        for movieid, item in self.probe_set.items():
            for consumerid, value in item.items():
                yield self.movieids[movieid], self.consumerids[
                    consumerid], value["rate"], value["date"]

    def convert(self, output):
        trainingfile = os.path.join(output, "trainingset.tfrecord")
        probefile = os.path.join(output, "probe.tfrecord")
        qualifyingfile = os.path.join(output, "qualifying.tfrecord")
        movieidsfile = os.path.join(output, "movieids.json")
        consumeridsfile = os.path.join(output, "consumerids.json")
        for movieindex, consumerindex, rate, date in self.load_training_set():
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "movieindex":
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[movieindex])),
                        "consumerindex":
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[consumerindex])),
                        "rate":
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[rate])),
                        "date":
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                [tf.compat.as_bytes(date)])),
                    }))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="netflix prize dataset converter")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    dataset = Dataset(directory=args.directory)
    dataset.convert(args.output)
