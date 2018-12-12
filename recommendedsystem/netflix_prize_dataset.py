import os
import glob
import tensorflow as tf
import json as JSON


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
            self.probe_set[movieid][consumerid] = {}

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
            if movieid in self.movieids and consumerid in self.consumerids:
                yield (self.movieids[movieid], self.consumerids[consumerid],
                       date)

    def load_probe_set(self):
        for movieid, item in self.probe_set.items():
            for consumerid, value in item.items():
                if value:
                    yield (self.movieids[movieid],
                           self.consumerids[consumerid], value["rate"],
                           value["date"])

    def tfexample(self, **kwargs):
        feature = {}
        if "movieindex" in kwargs:
            feature["movieindex"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[kwargs["movieindex"]]))
        if "consumerindex" in kwargs:
            feature["consumerindex"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[kwargs["consumerindex"]]))
        if "rate" in kwargs:
            feature["rate"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[kwargs["rate"]]))
        if "date" in kwargs:
            feature["date"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[tf.compat.as_bytes(kwargs["date"])]))
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def tfdateset(self, target="trainingset"):
        if target == "trainingset":
            filename = "trainingset.tfrecord"
        elif target == "qualifyingset":
            filename = "qualifying.tfrecord"
        elif target == "probeset":
            filename = "probe.tfrecord"
        else:
            raise ValueError("Unknown target: {}".format(target))

        def parse_single_example(example_proto):
            feature = tf.parse_single_example(
                example_proto, {
                    "movieindex":
                    tf.FixedLenFeature(
                        shape=[], dtype=tf.int64, default_value=0),
                    "consumerindex":
                    tf.FixedLenFeature(
                        shape=[], dtype=tf.int64, default_value=0),
                    "rate":
                    tf.FixedLenFeature(
                        shape=[], dtype=tf.float32, default_value=0),
                    "date":
                    tf.VarLenFeature(tf.string)
                })
            if "date" in feature:
                feature["date"] = tf.sparse_tensor_to_dense(
                    feature["date"], default_value="")
            return feature

        dataset = tf.data.TFRecordDataset(
            filenames=[os.path.join(self.directory, filename)],
            compression_type="GZIP")
        dataset = dataset.map(parse_single_example)
        return dataset

    def convert(self, output):
        trainingfile = os.path.join(output, "trainingset.tfrecord")
        probefile = os.path.join(output, "probe.tfrecord")
        qualifyingfile = os.path.join(output, "qualifying.tfrecord")
        movieidsfile = os.path.join(output, "movieids.json")
        consumeridsfile = os.path.join(output, "consumerids.json")
        with tf.python_io.TFRecordWriter(
                trainingfile,
                options=tf.python_io.TFRecordOptions(
                    compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        ) as writer:
            for (movieindex, consumerindex, rate,
                 date) in self.load_training_set():
                example = self.tfexample(
                    movieindex=movieindex,
                    consumerindex=consumerindex,
                    rate=rate,
                    date=date)
                writer.write(example.SerializeToString())
        with open(movieidsfile, "wt") as writer:
            content = JSON.dumps(self.movieids, ensure_ascii=False, indent=2)
            writer.writelines(content)
        with open(consumeridsfile, "wt") as writer:
            content = JSON.dumps(
                self.consumerids, ensure_ascii=False, indent=2)
            writer.writelines(content)
        with tf.python_io.TFRecordWriter(
                qualifyingfile,
                options=tf.python_io.TFRecordOptions(
                    compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        ) as writer:
            for movieindex, consumerindex, date in self.load_qualifying_set():
                example = self.tfexample(
                    movieindex=movieindex,
                    consumerindex=consumerindex,
                    date=date)
                writer.write(example.SerializeToString())
        with tf.python_io.TFRecordWriter(
                probefile,
                options=tf.python_io.TFRecordOptions(
                    compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        ) as writer:
            for (movieindex, consumerindex, rate,
                 date) in self.load_probe_set():
                example = self.tfexample(
                    movieindex=movieindex,
                    consumerindex=consumerindex,
                    rate=rate,
                    date=date)
                writer.write(example.SerializeToString())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="netflix prize dataset converter")
    subparsers = parser.add_subparsers()
    parser_convert = subparsers.add_parser(
        "convert", help="convert netflix prize dataset to tfrecord")
    parser_convert.set_defaults(command="convert")
    parser_convert.add_argument("--directory", required=True)
    parser_convert.add_argument("--output", required=True)

    parser_inspect = subparsers.add_parser(
        "inspect", help="inspect the converted tfrecord dataset")
    parser_inspect.set_defaults(command="inspect")
    parser_inspect.add_argument("--directory", required=True)
    parser_inspect.add_argument(
        "--set",
        choices=["training", "qualifying", "probe"],
        default="training")
    args = parser.parse_args()
    if args.command == "convert":
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        dataset = Dataset(directory=args.directory)
        dataset.convert(args.output)
    elif args.command == "inspect":
        dataset = Dataset(directory=args.directory)
        dataset = dataset.tfdateset("{}set".format(args.set))
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        tensors = iterator.get_next()
        init_ops = (tf.global_variables_initializer(),
                    tf.local_variables_initializer())
        tf.train.get_or_create_global_step()
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=None,
                hooks=[],
                save_checkpoint_secs=10,
                log_step_count_steps=100) as sess:
            sess.run(init_ops)
            index = 0
            try:
                while True:
                    results = sess.run(tensors)
                    print("{}: {}".format(index, results))
                    index += 1
            except tf.errors.OutOfRangeError:
                print("done")
    else:
        parser.error("Unknown command: {}".find(args.command))
