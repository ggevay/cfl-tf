import argparse
import sys

import tensorflow as tf




# i = tf.constant(0)
# coll = tf.constant([1,2,3])

# def cond(i, coll):
#   return tf.less(i, 10)

# def body(i, coll):
#   return (tf.add(i, 1), tf.add(coll, 1))

# r = tf.while_loop(cond, body, [i, coll])

# sess = tf.Session()
# print(sess.run(r))








# a = tf.constant([1,2])
# b = tf.constant([3,4])
# c = tf.concat([a,b], 0)

# sess = tf.Session()
# print(sess.run(c))








FLAGS = None


def main(_):
  loopmaster_host = [FLAGS.loopmaster_host]
  worker_hosts = FLAGS.worker_hosts.split(",")

  #print(loopmaster_host)
  #print(worker_hosts)

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"loopmaster": loopmaster_host, "worker": worker_hosts})

  print(cluster)

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)


  if FLAGS.job_name == "loopmaster":

    def cond(i, coll0, coll1):
      return tf.less(i, 10)

    def body(i, coll0, coll1):
      with tf.device("/job:worker/task:0"):
      #with tf.device("/job:loopmaster/task:0"):
        coll0_mapped = tf.add(coll0, 1)
      with tf.device("/job:worker/task:1"):
      #with tf.device("/job:loopmaster/task:0"):
        coll1_mapped = tf.add(coll1, 1)
      return (tf.add(i, 1), coll0_mapped, coll1_mapped)

    with tf.device("/job:loopmaster/task:0"):
      i = tf.constant(0)
      coll0_init = tf.constant([1,2])
      coll1_init = tf.constant([24,25])
      (i_res, coll0_res, coll1_res) = tf.while_loop(cond, body, [i, coll0_init, coll1_init])
      conc = tf.concat([coll0_res, coll1_res], 0)

    sess = tf.Session(server.target) # !
    print(sess.run(conc))

  elif FLAGS.job_name == "worker":
    server.join()


  writer = tf.summary.FileWriter('.')
  writer.add_graph(tf.get_default_graph())

  print("alma")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--loopmaster_host",
      type=str,
      default="",
      help="hostname:port pair"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'loopmaster', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)