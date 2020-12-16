import tensorflow as tf
import argparse
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
from preprocess import data_preprocessing
from model import make_generator_model, make_discriminator_model,\
                    generator_loss, discriminator_loss

if not os.path.exists('./fig'):
    os.mkdir('./fig')

def plot_loss(hist, path='./loss.png'):

    plt.close('all')

    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    # plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)

def generate_and_save_images(model, epoch, test_input):
  # `training`이 False로 맞춰진 것을 주목하세요.
  # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됩니다. 
  predictions = model(test_input, training=False)

  # fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('./fig/image_at_epoch_{:04d}.png'.format(epoch))
  plt.clf()
  # plt.show()

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='DCGAN')
    parser.add_argument('--alpha', default=1.0, type=float, help='one-sided label smoothing coefficient')
    parser.add_argument('--epoch', default=50, type=int, help='EPOCH')
    parser.add_argument('--batsize', default=256, type=int, help='BATCH SIZE')
    parser.add_argument('--bufsize', default=60000, type=int, help='BUFFER SIZE')
    parser.add_argument('--glr', default=1e-4, type=float, help='learning rate of G optimizer')
    parser.add_argument('--dlr', default=1e-4, type=float, help='learning rate of D optimizer')
    # parser.add_argument('--sum', help='')
    args = parser.parse_args()
    print(args)

    train_dataset, train_labels, _, _=data_preprocessing(args.bufsize, args.batsize)
    generator=make_generator_model()
    discriminator=make_discriminator_model()

    # D and G learns seperately
    generator_optimizer = tf.keras.optimizers.Adam(args.glr)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.dlr)

    # save checkpoint
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    # train_hist['per_epoch_ptimes'] = []
    # train_hist['total_ptime'] = []

    ## Training Loop
    noise_dim = 100
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # `tf.function`이 어떻게 사용되는지 주목해 주세요.
    # 이 데코레이터는 함수를 "컴파일"합니다.
    # @tf.function
    def train_step(images):
        noise = tf.random.normal([args.batsize, noise_dim])

        # D and G learns separately
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output, args.alpha)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        return gen_loss.numpy(), disc_loss.numpy()

    def train(dataset, epochs):

        for epoch in range(epochs):
            start = time.time()

            gloss_per_batch=[]; dloss_per_batch=[]

            for image_batch in dataset:
                G_loss, D_loss = train_step(image_batch)
                gloss_per_batch.append(G_loss)
                dloss_per_batch.append(D_loss)

            print('[EPOCH {}] G_loss: {}, D_loss: {}'.format(epoch, np.mean(gloss_per_batch), np.mean(dloss_per_batch)))
            train_hist['G_losses'].append(np.mean(gloss_per_batch))
            train_hist['D_losses'].append(np.mean(dloss_per_batch))

            # GIF를 위한 이미지를 바로 생성합니다.
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                    epoch + 1,
                                    seed)

            # 15 에포크가 지날 때마다 모델을 저장합니다.
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            
            # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # 마지막 에포크가 끝난 후 생성합니다.
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epochs,
                                seed)

        display.clear_output(wait=True)
        plot_loss(train_hist)
    train(train_dataset, args.epoch)