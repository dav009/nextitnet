import tensorflow as tf
import data_loader_recsys
import generator_recsys
import utils
import shutil
import time
import math
import eval
import json
import sys
import numpy as np
import argparse

# You can run it directly, first training and then evaluating
# nextitrec_generate.py can only be run when the model parameters are saved, i.e.,
#  save_path = saver.save(sess,
#                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))
# if you are dealing very huge industry dataset, e.g.,several hundred million items, you may have memory problem during training, but it 
# be easily solved by simply changing the last layer, you do not need to calculate the cross entropy loss
# based on the whole item vector. Similarly, you can also change the last layer (use tf.nn.embedding_lookup or gather) in the prediction phrase 
# if you want to just rank the recalled items instead of all items. The current code should be okay if the item size < 5 million.



#Strongly suggest running codes on GPU with more than 10G memory!!!
#if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
def generatesubsequence(train_set):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        # print x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        for j in range(lenseq - 2):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [0] * j
            subseq = np.append(subseqbeg, subseqend)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)
    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print "generating subsessions is done!"
    return x_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=10,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    #this is a demo dataset, which just let you run this code, suggest dataset link: http://grouplens.org/datasets/.
    parser.add_argument('--datapath', type=str, default='Data/Session/',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=100,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=100,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument("--iterations", type=int, default=2, help='number of training iterations')
    parser.add_argument("--dilated_channels", type=int, default=100, help='number of dilated channels')
    parser.add_argument("--learning_rate", type=float, default=0.008, help='learning rate')
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
    parser.add_argument("--batch_size", type=int, default=300, help="batch size")
    parser.add_argument("--max_seq_size", type=int, default=80, help="max seq len")
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    args = parser.parse_args()

    training_path = args.datapath + "/" + "training.csv"
    model_path = args.datapath + "/" + "model.ckpt"
    vocab_path = args.datapath + "/" + "vocab.pickle"

    dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': training_path}, max_seq_size=args.max_seq_size)
    all_samples = dl.item
    items = dl.item_dict
    print("len(items)")
    print(len(all_samples))

    with open(vocab_path, 'w') as fp:
        json.dump(items, fp)

    with open(vocab_path+"inverted", 'w') as fp:
        json.dump(dl.vocabulary, fp)
 

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]


    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    if args.is_generatesubsession:
        x_train = generatesubsequence(train_set)

    model_para = {
        #if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
        'item_size': len(items)+ 1,
        'dilated_channels': args.dilated_channels,#larger is better until 512 or 1024
        # if you use nextitnet_residual_block, you can use [1, 4, 1, 4, 1,4,],
        # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
        # when you change it do not forget to change it in nextitrec_generate.py
        'dilations': [1, 2, 1, 2, 1, 2, ],#YOU should tune this hyper-parameter, refer to the paper.
        'kernel_size': args.kernel_size,
        'learning_rate': args.learning_rate,#YOU should tune this hyper-parameter
        'batch_size': args.batch_size,#YOU should tune this hyper-parameter
        'iterations': args.iterations,# if your dataset is small, suggest adding regularization to prevent overfitting probably bump this to 100
        'is_negsample':False #False denotes no negative sampling
    }

    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'])
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)
    itemrec.predict_graph(model_para['is_negsample'],reuse=True)

    sess= tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    numIters = 1
    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        while (batch_no + 1) * batch_size < train_set.shape[0]:

            start = time.clock()

            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]
            _, loss, results = sess.run(
                [optimizer, itemrec.loss,
                 itemrec.arg_max_prediction],
                feed_dict={
                    itemrec.itemseq_input: item_batch
                })
            end = time.clock()
            if numIters % args.eval_iter == 0:
                print "-------------------------------------------------------train1"
                print "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter, batch_no, numIters, train_set.shape[0] / batch_size)
                print "TIME FOR BATCH", end - start
                print "TIME FOR ITER (mins)", (end - start) * (train_set.shape[0] / batch_size) / 60.0

            if numIters % args.eval_iter == 0:
                print "-------------------------------------------------------test1"
                if (batch_no + 1) * batch_size < valid_set.shape[0]:
                    item_batch = valid_set[(batch_no) * batch_size: (batch_no + 1) * batch_size, :]
                loss = sess.run(
                    [itemrec.loss_test],
                    feed_dict={
                        itemrec.input_predict: item_batch
                    })
                print "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter, batch_no, numIters, valid_set.shape[0] / batch_size)

            batch_no += 1


            if numIters % args.eval_iter == 0:
                batch_no_test = 0
                batch_size_test = batch_size*1
                curr_preds_5=[]
                rec_preds_5=[] #1
                ndcg_preds_5=[] #1
                curr_preds_20 = []
                rec_preds_20 = []  # 1
                ndcg_preds_20  = []  # 1
                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    if (numIters / (args.eval_iter) < 10):
                        if (batch_no_test > 20):
                            break
                    else:
                        if (batch_no_test > 500):
                            break
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    [probs] = sess.run(
                        [itemrec.g_probs],
                        feed_dict={
                            itemrec.input_predict: item_batch
                        })
                    for bi in range(probs.shape[0]):
                        pred_items_5 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k)#top_k=5
                        pred_items_20 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k+15)


                        true_item=item_batch[bi][-1]
                        predictmap_5={ch : i for i, ch in enumerate(pred_items_5)}
                        pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

                        rank_5=predictmap_5.get(true_item)
                        rank_20 = pred_items_20.get(true_item)
                        if rank_5 ==None:
                            curr_preds_5.append(0.0)
                            rec_preds_5.append(0.0)#2
                            ndcg_preds_5.append(0.0)#2
                        else:
                            MRR_5 = 1.0/(rank_5+1)
                            Rec_5=1.0#3
                            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
                            curr_preds_5.append(MRR_5)
                            rec_preds_5.append(Rec_5)#4
                            ndcg_preds_5.append(ndcg_5)  # 4
                        if rank_20 ==None:
                            curr_preds_20.append(0.0)
                            rec_preds_20.append(0.0)#2
                            ndcg_preds_20.append(0.0)#2
                        else:
                            MRR_20 = 1.0/(rank_20+1)
                            Rec_20=1.0#3
                            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
                            curr_preds_20.append(MRR_20)
                            rec_preds_20.append(Rec_20)#4
                            ndcg_preds_20.append(ndcg_20)  # 4


                    batch_no_test += 1
                    print "BATCH_NO: {}".format(batch_no_test)
                    print "Accuracy mrr_5:",sum(curr_preds_5) / float(len(curr_preds_5))#5
                    print "Accuracy mrr_20:", sum(curr_preds_20) / float(len(curr_preds_20))  # 5
                    print "Accuracy hit_5:", sum(rec_preds_5) / float(len(rec_preds_5))#5
                    print "Accuracy hit_20:", sum(rec_preds_20) / float(len(rec_preds_20))  # 5
                    print "Accuracy ndcg_5:", sum(ndcg_preds_5) / float(len(ndcg_preds_5))  # 5
                    print "Accuracy ndcg_20:", sum(ndcg_preds_20) / float(len(ndcg_preds_20))  #
                    #print "curr_preds",curr_preds
                # print "---------------------------Test Accuray----------------------------"
            numIters += 1
            if numIters % args.save_para_every == 0:
                print("saving..")
                save_path = saver.save(sess, model_path)


if __name__ == '__main__':
    main()
