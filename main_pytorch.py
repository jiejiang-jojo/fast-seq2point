import argparse
import numpy as np
import os
import time
import json
#import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging,
                       mean_absolute_error, signal_aggregate_error)
from data_generator import DataGenerator, TestDataGenerator
from models import move_data_to_gpu
from models import *

def loss_func(output, target):

    assert output.shape == target.shape

    return torch.mean(torch.abs(output - target))


def evaluate(model, generator, data_type, max_iteration, cuda):
    """Evaluate.

    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      max_iteration: int, maximum iteration for validation
      cuda: bool.

    Returns:
      mae: float
    """

    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                max_iteration=max_iteration)

    # Forward
    (outputs, targets) = forward(model=model,
                                 generate_func=generate_func,
                                 cuda=cuda,
                                 has_target=True)

    outputs = generator.inverse_transform(outputs)
    targets = generator.inverse_transform(targets)

    mae = mean_absolute_error(outputs, targets)

    return mae


def forward(model, generate_func, cuda, has_target):
    """Forward data to a model.

    Args:
      model: object
      generate_func: generate function
      cuda: bool
      has_target: bool, True if generate_func yield (batch_x, batch_y),
                        False if generate_func yield (batch_x)

    Returns:
      (outputs, targets) | outputs
    """

    model.eval()

    outputs = []
    targets = []

    # Evaluate on mini-batch
    for data in generate_func:

        if has_target:
            (batch_x, batch_y) = data
            targets.append(batch_y)

        else:
            batch_x = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        batch_output = model(batch_x)

        outputs.append(batch_output.data.cpu().numpy())

    if has_target:
        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        return outputs, targets

    else:

        return outputs


def train(args):

    logging.info('config=%s', json.dumps(vars(args)))

    # Arguments & parameters
    workspace = args.workspace
    cuda = args.cuda

    # Model
    model_class, model_params = MODELS[args.model]
    model = model_class(**{k: args.model_params[k] for k in model_params if k in args.model_params})
    logging.info("sequence length: {}".format(model.seq_len))

    if cuda:
        model.cuda()

    # Paths
    hdf5_path = os.path.join(workspace, 'data.h5')

    models_dir = os.path.join(workspace, 'models', get_filename(__file__))

    create_folder(models_dir)

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              target_device=args.target_device,
                              train_house_list=args.train_house_list,
                              validate_house_list=args.validate_house_list,
                              batch_size=args.batch_size,
                              seq_len=model.seq_len,
                              width=args.width)

    # Optimizer
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    iteration = 0
    train_bgn_time = time.time()

    for (batch_x, batch_y) in generator.generate():

        if iteration > 1000*50:
           break

        # Evaluate
        if iteration % 1000 == 0:

            train_fin_time = time.time()

            tr_mae = evaluate(model=model,
                              generator=generator,
                              data_type='train',
                              max_iteration=args.validate_max_iteration,
                              cuda=cuda)

            va_mae = evaluate(model=model,
                              generator=generator,
                              data_type='validate',
                              max_iteration=args.validate_max_iteration,
                              cuda=cuda)

            logging.info('tr_mae: {:.4f}, va_mae: {:.4f}'.format(
                tr_mae, va_mae))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'.format(
                    iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Reduce learning rate
        if iteration % 1000 == 0 and iteration > 0 and learning_rate > 5e-5:
            for param_group in optimizer.param_groups:
                learning_rate *= 0.9
                param_group['lr'] = learning_rate

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        # Forward
        forward_time = time.time()
        model.train()
        output = model(batch_x)

        # Loss
        loss = loss_func(output, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save model
        if (iteration>1) and (iteration % 10000 == 0):
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}

            save_out_path = args.basename + '_{}_{}_iter_{}.tar'.format(
                args.target_device,
                args.model,
                iteration
            )

            create_folder(os.path.dirname(save_out_path))
            torch.save(save_out_dict, save_out_path)

            print('Save model to {}'.format(save_out_path))

        iteration += 1


def inference(args):

    logging.info('config=%s', json.dumps(vars(args)))
    # Arguments & parameters
    workspace = args.workspace
    iteration = args.iteration
    cuda = args.cuda

    # Paths
    hdf5_path = os.path.join(workspace, 'data.h5')
    model_path = os.path.join(workspace, 'models', get_filename(__file__),
                              args.inference_model)
    # model_path = os.path.join(workspace, 'models', get_filename(__file__), 'microwaveWaveNet2all_md_50000_iters.tar')

    # Load model
    model_class, model_params = MODELS[args.model]
    model = model_class(**{k: args.model_params[k] for k in model_params if k in args.model_params})
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = TestDataGenerator(hdf5_path=hdf5_path,
                                target_device=args.target_device,
                                train_house_list=args.train_house_list,
                                seq_len=model.seq_len,
                                steps=args.width * args.batch_size)

    generate_func = generator.generate_inference(house=args.inference_house)

    # Forward
    inference_time = time.time()

    outputs = forward(model=model, generate_func=generate_func, cuda=cuda, has_target=False)
    outputs = np.concatenate([output[0] for output in outputs])
    outputs = generator.inverse_transform(outputs)

    print('Inference time: {} s'.format(time.time() - inference_time))

    # Calculate metrics
    source = generator.get_source()
    targets = generator.get_target()

    valid_data = np.ones_like(source)
    for i in range(len(source)):
        if (source[i]==0) or (source[i] < targets[i]):
            valid_data[i] = 0

    mae = mean_absolute_error(outputs * valid_data, targets * valid_data)
    sae = signal_aggregate_error(outputs * valid_data, targets * valid_data)
    mae_allzero = mean_absolute_error(outputs*0, targets * valid_data)
    sae_allmean = signal_aggregate_error(outputs*0+np.mean(targets), targets * valid_data)

    print('MAE: {}'.format(mae))
    print('MAE all zero: {}'.format(mae_allzero))
    print('SAE: {}'.format(sae))
    print('SAE all mean: {}'.format(sae_allmean))

    np.save(workspace+'outputs/'+args.inference_model+'_'+args.inference_house+'_'+'prediction.npy', outputs)
    np.save(workspace+'outputs/'+args.inference_model+'_'+args.inference_house+'_'+'groundtruth.npy', targets)


def consolidate_args(args):
    """ Merge different source of configuration. """
    # Loading config into args
    with open(args.config) as fin:
        config = json.load(fin)
        args.__dict__.update(config)
    # Loading commandline model parameters into model_params
    model_param_setting = {k: v for k, v in args.__dict__.items() if k.startswith('pm_')}
    if 'model_params' not in args.__dict__:
        args.__dict__['model_params'] = {}
    for k, v in model_param_setting.items():
        args.__dict__.pop(k)
        if v is not None:
            try:
                args.model_params[k[3:].replace('-', '_')] = int(v)
            except:
                args.model_params[k[3:]] = v


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    model_params = set()
    for _, (_, mps) in MODELS.items():
        for p in mps:
            model_params.add(p)

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--config', type=str, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    for p in model_params:
        parser_train.add_argument('--pm-' + p.replace('_', '-'), type=str, metavar='<{}>'.format(p))

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--config', type=str, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    consolidate_args(args)

    # Write out log
    logs_dir = os.path.join(args.workspace, 'logs', get_filename(__file__))
    logging = create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        args.__dict__['basename'] = logging.getLoggerClass().root.handlers[0].baseFilename[:-4]
        config_to_save = args.basename + '.config.json'
        logging.info('Saving config to ' + config_to_save)
        ignores = set(['workspace', 'config', 'cuda', 'mode'])
        with open(config_to_save, 'w') as fout:
            json.dump({k: v for k, v in args.__dict__.items()
                       if k not in ignores}, fout)
        train(args)

    elif args.mode == 'inference':
        inference(args)

    else:
        raise Exception('Error!')
