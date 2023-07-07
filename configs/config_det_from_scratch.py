# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Config file for object detection training from scratch."""

import copy
# pylint: disable=invalid-name,line-too-long,missing-docstring
from configs.config_base import architecture_config_map
from configs import dataset_configs
from configs import transform_configs
from configs.config_base import architecture_config_map
from configs.config_base import D

task_specific_coco_dataset_config = {
    'object_detection@coco/2017_object_detection':
        D(
            name='object_detection',
            tfds_name='coco/2017',
            train_filename='instances_train2017.json',
            val_filename='instances_val2017.json',
        ),
}


def get_config(config_str=None):
  """config_str is either empty or contains task,architecture variants."""

  task_variant = 'object_detection@coco/2017_object_detection'
  encoder_variant = 'resnet'                 # Set model architecture.
  image_size = [1333, 1333]                          # Set image size.

  tasks_and_datasets = []
  for task_and_ds in task_variant.split('+'):
    tasks_and_datasets.append(task_and_ds.split('@'))

  max_instances_per_image = 100
  max_instances_per_image_test = 100
  task_config_map = {
      'object_detection': D(
          name='object_detection',
          vocab_id=10,
          image_size=image_size,
          quantization_bins=2000,
          max_instances_per_image=max_instances_per_image,
          max_instances_per_image_test=max_instances_per_image_test,
        #   object_order='random',
        #   color_jitter_strength=0.,
        #   jitter_scale_min=0.1,
        #   jitter_scale_max=3.0,
          train_transforms=transform_configs.get_object_detection_train_transforms(
              image_size, max_instances_per_image, jitter_scale_min=0.1, jitter_scale_max=3.0),
          eval_transforms=transform_configs.get_object_detection_eval_transforms(
              image_size, max_instances_per_image_test),
          # Train on both ground-truth and noisy objects.
          noise_bbox_weight=1.0,
          eos_token_weight=0.1,
          # Train on just ground-truth objects (and ending token).
          # noise_bbox_weight=0.0,
          # eos_token_weight=0.1,    # set to 0 for no ending token
          class_label_corruption='rand_n_fake_cls',
          top_k=0,
          top_p=0.3,
          temperature=1.0,
          weight=1.0,
          metric=D(name='coco_object_detection',),
      ),
  }

  task_d_list = []
  dataset_list = []
  for tv, ds_name in tasks_and_datasets:
    task_d_list.append(task_config_map[tv])
    dataset_config = copy.deepcopy(dataset_configs.dataset_configs[ds_name])
    dataset_list.append(dataset_config)

  config = D(
      dataset=dataset_list[0],
      datasets=dataset_list,

      task=task_d_list[0],
      tasks=task_d_list,

      model=D(
          name='encoder_ar_decoder',
          image_size=image_size,
          max_seq_len=512,
          vocab_size=3000,                  # Note: should be large enough for 100 + num_classes + quantization_bins + (optional) text
          coord_vocab_shift=1000,           # Note: make sure num_class <= coord_vocab_shift - 100
          text_vocab_shift=3000,            # Note: make sure coord_vocab_shift + quantization_bins <= text_vocab_shift
          use_cls_token=False,
          shared_decoder_embedding=True,
          decoder_output_bias=True,
          patch_size=16,
          drop_path=0.1,
          drop_units=0.1,
          drop_att=0.,
          dec_proj_mode='mlp',
          pos_encoding='sin_cos',
          pos_encoding_dec='learned',
          pretrained_ckpt='',
      ),

      optimization=D(
          optimizer='adamw',
          learning_rate=3e-3,
          end_lr_factor=0.01,
          warmup_epochs=10,
          warmup_steps=0,
          weight_decay=0.05,
          global_clipnorm=-1,
          beta1=0.9,
          beta2=0.95,
          eps=1e-8,
          learning_rate_schedule='linear',
          learning_rate_scaling='none',
      ),

      train=D(
          batch_size=128,
          epochs=300,
          steps=0,
          checkpoint_epochs=1,
          checkpoint_steps=0,
          keep_checkpoint_max=5,
          loss_type='xent',
      ),

      eval=D(
          tag='eval',
          checkpoint_dir='',
          batch_size=8,
          steps=0,
      ),
  )

  # Update model with architecture variant.
  for key, value in architecture_config_map[encoder_variant].items():
    config.model[key] = value

  return config