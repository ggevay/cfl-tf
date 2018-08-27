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
  num_hosts = len(worker_hosts)

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

    def cond(i, *colls):
      return tf.less(i, 10)

    def body(i, *colls):
      t = 0
      coll_mapped = []
      for coll in colls:
        with tf.device("/job:worker/task:%d" % t):
        #with tf.device("/job:loopmaster/task:0"):
          coll_mapped.append(tf.add(coll, 1))
      return tuple([tf.add(i, 1)] + coll_mapped)

    with tf.device("/job:loopmaster/task:0"):
      i = tf.constant(0)
      # coll0_init = tf.constant([1,2])
      # coll1_init = tf.constant([24,25])
      coll_init = []
      for j in range(0,num_hosts):
        coll_init.append(tf.constant([j]))
      res = tf.while_loop(cond, body, [i] + coll_init)
      i_res = res[0]
      colls_res = res[1:num_hosts+1]
      conc = tf.concat(colls_res, 0)

    sess = tf.Session(server.target) # !
    print(sess.run(conc))

  elif FLAGS.job_name == "worker":
    server.join()


  writer = tf.summary.FileWriter('.')
  writer.add_graph(tf.get_default_graph())


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