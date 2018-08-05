# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys
import math
# reload(sys)
# sys.setdefaultencoding('utf8')
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
#from p1_HierarchicalAttention_model import HierarchicalAttention
#from soft_entropy_dynamic_addlayer import HierarchicalAttention
from soft_loss import HierarchicalAttention
#from p1_HierarchicalAttention_model_transformer import HierarchicalAttention

from tflearn.data_utils import to_categorical, pad_sequences
import os
import word2vec
import pickle
from My_word2vec_Glove import *

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",5,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate") #TODO 0.01
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.") #批处理的大小 32-->128 #TODO
tf.app.flags.DEFINE_integer("decay_steps", 2000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate",0.9 , "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","HAN/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",1250,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")
tf.app.flags.DEFINE_integer("is_training",1,"1:training,2:join,3:test")
tf.app.flags.DEFINE_integer("num_epochs",60,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_integer("validate_step", 400, "how many step to validate.") #1500做一次检验 TODO
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
#train-zhihu4-only-title-all.txt
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.")
tf.app.flags.DEFINE_integer("num_sentences", 25, "number of sentences in the document") #每10轮做一次验证
tf.app.flags.DEFINE_integer("hidden_size",50,"hidden size")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #1.load data(X:list of lint,y:int).
    #if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    #else:
    if 1==1:
        vocabulary_word2index, vocabulary_index2word = {}, {}
        trainX, trainY, validX, validY, testX, testY= read_yelp(path="../dataset",year=2013)
        print(trainX[0])
        print(validX[0])
        cache_path='../dataset/cache_vocabulary_label_pik/'+"yelp_2013_glove_eospad_double_200d.5allwords.pik"
        if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
            with open(cache_path, 'rb') as data_f:
                embeddings,vocabulary_word2index,vocabulary_index2word = pickle.load(data_f, encoding='latin1')

        if "<PAD>" not in vocabulary_word2index:
            bound = np.sqrt(6.0) / np.sqrt(len(vocabulary_word2index))
            vocabulary_word2index["<PAD>"]=len(vocabulary_word2index)
            vocabulary_index2word[len(vocabulary_index2word)]="<PAD>"
            embeddings = np.append(embeddings,np.random.uniform(-bound, bound,
                        FLAGS.embed_size).astype(np.float32)).reshape(-1,FLAGS.embed_size)
        print("<PAD> index:", vocabulary_word2index['<PAD>'])
        vocab_size = len(vocabulary_word2index)
        print("cnn_model.vocab_size:",vocab_size)
        
        print("testX.shape:", np.array(testX).shape)  
        
        print("testY.shape:", np.array(testY).shape) 
        

        print("trainX[0]:", trainX[0])

        print("trainX[1820]:", trainX[1820])

        trainX,validX,testX = transform_text(trainX,validX,testX,vocabulary_word2index)
        
        print("testX.shape:", np.array(testX).shape)  
 i       print("testY.shape:", np.array(testY).shape)
        print("testX[0]:", testX[0]) 
        print("testY[0]:", testY[0]) 

        
        #raw_input("continue")

        # 2.Data preprocessing.Sequence padding
        print("start padding & transform to one hot...")
        trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=vocabulary_word2index["<PAD>"])  # padding to max length
        validX = pad_sequences(validX, maxlen=FLAGS.sequence_length, value=vocabulary_word2index["<PAD>"])  # padding to max length
        testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=vocabulary_word2index["<PAD>"])  # padding to max length
        #with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
        #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
        print("trainX[0]:", trainX[0]) #;print("trainY[0]:", trainY[0])
        # Converting labels to binary vectors
        print("end padding & transform to one hot...")

        print("start calculating tfidf score...")
        len_train, len_valid, len_test = len(trainX), len(validX), len(testX)
        print(type(trainX))
        print('shape:', trainX.shape, validX.shape, testX.shape)
        content = np.concatenate((trainX, validX, testX), axis=0)
        print('content.shape:', content.shape)
        str_content = []
        for i in content:
            str_content.append(list(map(str, i)))
        tf_idf_fea = tf_idf(str_content)
        tf_train, tf_valid, tf_test = tf_idf_fea[:len_train], tf_idf_fea[len_train: len_train+len_valid], tf_idf_fea[len_train+len_valid:]
        print('tf_train shape', tf_train.shape)
        print('tf_valid.shape', tf_valid.shape)
        print('tf_test.shape', tf_test.shape)
        print('tf_train[0]:', tf_train[0])

        print("end tfidf")

    #2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        #Instantiate Model
        #num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,num_sentences,vocab_size,embed_size,
        #hidden_size,is_training
        model=HierarchicalAttention(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.sequence_length,
                                       FLAGS.num_sentences,vocab_size,FLAGS.embed_size,FLAGS.hidden_size,FLAGS.is_training,multi_label_flag=FLAGS.multi_label_flag)
        
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, embeddings,model)
        curr_epoch=sess.run(model.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        print("number_of_training_data:",number_of_training_data)
        previous_eval_loss=1000000
        best_eval_loss=1000000
        batch_size=FLAGS.batch_size
        # tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("HAN/train", sess.graph)
        # embedding visualize
        '''
        v_config = projector.ProjectorConfig()   
        v_embedding_var = tf.Variable(tf.transpose(model.att_word),name='V_Embedding')
        v_embedding = v_config.embeddings.add()
        PATH_TO_SPRITE_IMAGE = os.path.join("HAN", 'visual_embed.png')  
        v_embedding.tensor_name = v_embedding_var.name
        v_embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE 
        v_embedding.sprite.single_image_dim.extend([28, 28])
        embed_writer = tf.summary.FileWriter("HAN")
        projector.visualize_embeddings(embed_writer, v_config)
        '''
        # count_y
        count_y = np.zeros([FLAGS.num_classes,1])
        for i in range(FLAGS.num_classes):
            count_y[i] = trainY.count(i)
        count_y = count_y/np.sum(count_y)
        count_y = 0.5 - count_y/FLAGS.num_classes
        t_count_y = tf.assign(model.alpha_y,count_y)
        sess.run(t_count_y)
        print("count_y:",count_y)

        
        global test_counter
        test_counter = 0

        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])#;print("trainY[start:end]:",trainY[start:end])
                feed_dict = {model.input_x: trainX[start:end],model.dropout_keep_prob: 0.5}
                if not FLAGS.multi_label_flag:
                    feed_dict[model.input_y] = trainY[start:end]
                else:
                    feed_dict[model.input_y_multilabel]=trainY[start:end]
                
                # tensorboard
                #summary , curr_loss,curr_acc,_=sess.run([merged, model.loss_val,model.accuracy,model.train_op],feed_dict) #curr_acc--->TextCNN.accuracy
                gd_sigma,sigma,sigma1,see_att,worda, sena, summary,curr_loss,curr_acc,_=sess.run([model.gradient,model.sigma,model.sigma1,model.see_attention, model.context_vecotor_word,model.context_vecotor_sentence , merged, model.loss_val,model.accuracy,model.train_op],feed_dict) #curr_acc--->TextCNN.accuracy
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                
                # tensorboard
                train_writer.add_summary(summary,
                        counter+epoch*((number_of_training_data-1)/batch_size))
                
                if counter % 2==0:
                    print("sigma:",sigma," gradient:",gd_sigma)
                    print("sigma1:",sigma1)
                    #print("see_att:",see_att)
                    #print("word:",worda)
                    #print("sen:",sena)
                    # word attention
                    #print("word sim:",np.sum(worda[0]*worda[1]))
                    #print("word sim:",np.sum(worda[0]*worda[0]))
                    #print("word sim:",np.sum(worda[1]*worda[1]))
                    #print("word sim:",np.sum(worda[0]*worda[1])/math.sqrt(np.sum(worda[0]*worda[0])*np.sum(worda[1]*worda[1])))
                    # sentence attention
                    #print("sen sim:",np.sum(sena[0]*sena[1]))
                    #print("sen sim:",np.sum(sena[0]*sena[0]))
                    #print("sen sim:",np.sum(sena[1]*sena[1]))
                    #print("sen sim:",np.sum(sena[0]*sena[1])/math.sqrt(np.sum(sena[0]*sena[0])*np.sum(sena[1]*sena[1])))
                    print("HierAtten_0609drate0.75==>Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)

                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################
                if FLAGS.batch_size!=0 and (start%(FLAGS.validate_step*FLAGS.batch_size)==0): #(epoch % FLAGS.validate_every) or  if epoch % FLAGS.validate_every == 0:
                    eval_loss, eval_acc = do_eval(sess, model, testX, testY, batch_size)
                    print("validation.part. previous_eval_loss:",
                            previous_eval_loss,";current_eval_loss:", eval_loss ," ;Validation Accuracy: %.3f" % (eval_acc))
                    if eval_loss > previous_eval_loss: #if loss is not decreasing
                        # reduce the learning rate by a factor of 0.5
                        print("HierAtten_0609drate0.75==>validation.part.going to reduce the learning rate.")
                        learning_rate1 = sess.run(model.learning_rate)
                        lrr=sess.run([model.learning_rate_decay_half_op])
                        learning_rate2 = sess.run(model.learning_rate)
                        print("HierAtten_0609drate0.75==>validation.part.learning_rate1:", learning_rate1, " ;learning_rate2:",learning_rate2)
                    #print("HierAtten==>Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                    else:# loss is decreasing
                        if eval_loss<best_eval_loss:
                            print("HierAtten_0609drate0.75==>going to save the model.eval_loss:",eval_loss,";best_eval_loss:",best_eval_loss)
                            # save model to checkpoint
                            save_path = FLAGS.ckpt_dir + "model.ckpt"
                            saver.save(sess, save_path, global_step=epoch)
                            best_eval_loss=eval_loss
                    previous_eval_loss = eval_loss
                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################

                if counter % 50==0:
                    print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
                    if epoch % FLAGS.validate_every==0:
                        eval_loss, eval_acc=do_eval(sess,model,testX,testY,batch_size)
                        print("HierAtten==>Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_acc))

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss, eval_acc=do_eval(sess,model,testX,testY,batch_size)
                print("HierAtten==>Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_acc))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, model, testX, testY, batch_size)
    pass


def transform_text(X,X1,X2,word2index):
    trainX,validX,testX=[],[],[]
    for sentence in X:
        now=[]
        for word in sentence:
            if word in word2index:
                now.append(word2index[word])
            else:
                now.append(word2index['<UNK>'])
        trainX.append(now)
    for sentence in X1:
        now=[]
        for word in sentence:
            if word in word2index:
                now.append(word2index[word])
            else:
                now.append(word2index['<UNK>'])
        validX.append(now)
    for sentence in X2:
        now=[]
        for word in sentence:
            if word in word2index:
                now.append(word2index[word])
            else:
                now.append(word2index['<UNK>'])
        testX.append(now)       
    return trainX,validX,testX



def assign_pretrained_word_embedding(sess,word_embedding_final,textRCNN):
    print("using pre-trained word emebedding.started...")
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)
    t_assign_embedding = tf.assign(textRCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    
    soft_label = np.array([range(FLAGS.num_classes)])
    t_soft_label = tf.assign(textRCNN.soft_label,soft_label)
    sess.run(t_soft_label)

    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度
def do_eval(sess,textCNN,evalX,evalY,batch_size):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    global test_counter
    # tensorboard
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter("HAN/test")
    see_ac = []

    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        
        
        # tensorboard
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1}
        if not FLAGS.multi_label_flag:
            feed_dict[textCNN.input_y] = evalY[start:end]
        else:
            feed_dict[textCNN.input_y_multilabel] = evalY[start:end]
        #loss1,loss2,loss3,loss4,loss5,soft_logit,soft_y,summary,curr_eval_loss,logits,curr_eval_acc=sess.run([textCNN.loss,textCNN.soft_loss,textCNN.final_loss,textCNN.l2_losses,textCNN.l2_soft_loss, textCNN.soft_logits,textCNN.soft_y,merged,textCNN.loss_val,textCNN.logits,textCNN.accuracy],feed_dict)#curr_eval_acc--->textCNN.accuracy
        loss1,loss3,loss4,summary,curr_eval_loss,logits,curr_eval_acc=sess.run([textCNN.loss,textCNN.final_loss,textCNN.l2_losses,merged,textCNN.loss_val,textCNN.logits,textCNN.accuracy],feed_dict)#curr_eval_acc--->textCNN.accuracy
        print("loss:",loss1)
        #print("soft_loss:",loss2)
        print("final_loss:",loss3)
        print("l2_losses:",loss4)
        #print("l2_soft_losses:",loss5)


        #np.set_printoptions(threshold='nan')
        #print(sess.run([textCNN.output_logit],feed_dict))
        #print(sess.run([textCNN.W_P,textCNN.W_G],feed_dict))

        #print("soft_y:",soft_y)
        #print("soft_logit:",np.exp(soft_logit)/np.sum(np.exp(soft_logit),axis=1,keepdims=True))
        # tensorboard
        see_ac.append(curr_eval_acc)
        test_counter = test_counter + 1
        test_writer.add_summary(summary, test_counter)
        
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        #eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss*(end-start+1),eval_acc+curr_eval_acc*(end-start+1),eval_counter+(end-start+1)
    #return eval_loss/float(number_examples),eval_acc/float(number_examples)
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    print("more information:",see_ac)
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

#从logits中取出前五 get label using logits
def get_label_using_logits(logits,top_number=1):
    #print("get_label_using_logits.logits:",logits) #1-d array: array([-5.69036102, -8.54903221, -5.63954401, ..., -5.83969498,-5.84496021, -6.13911009], dtype=float32))
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    #label_list=[]
    #for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

if __name__ == "__main__":
    tf.app.run()
