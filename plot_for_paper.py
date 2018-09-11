import argparse
import numpy as np
import matplotlib.pyplot as plt


def mae():
    
    metric = 'mae'
    
    fig, axs = plt.subplots(2,2, figsize=(7, 7))
    
    # Kettle
    if metric == 'mae':
        cnn_mae = np.array([10.194657, 10.194615, 7.464008, 10.259135, 10.336272, 6.849025])
        rnn_mae = np.array([5.124155, 5.548042, 5.165635, 4.817543, 5.514688, 4.630165])
        wavenet_mae = np.array([10.186407, 5.383598, 5.237763, 4.672643, 5.021998, 5.123513, 5.620855, 6.190022])
    elif metric == 'sae':
        cnn_mae = np.array([0.994668, 1.005243, 0.582693, 1.014465, 1.025485, 0.470792])
        rnn_mae = np.array([0.149725, 0.168050, 0.210650, 0.285975, 0.214575, 0.264550])
        wavenet_mae = np.array([1.004072, 0.287530, 0.250958, 0.202195, 0.191095, 0.222707, 0.333962, 0.399807])
    
    tick_labels = ['15', '31', '63', '127', '255', '511', '1023', '2047']

    axs[0, 0].scatter(np.arange(len(cnn_mae)), cnn_mae, color='g')
    axs[0, 0].scatter(np.arange(len(rnn_mae)), rnn_mae, color='b')
    axs[0, 0].scatter(np.arange(len(wavenet_mae)), wavenet_mae, color='r')
    
    line_cnn, = axs[0, 0].plot(cnn_mae, color='g')
    line_rnn, = axs[0, 0].plot(rnn_mae, color='b')
    line_wavenet, = axs[0, 0].plot(wavenet_mae, color='r')
    axs[0, 0].legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs[0, 0].set_title('Kettle')
    axs[0, 0].xaxis.set_ticks(np.arange(len(tick_labels)))
    axs[0, 0].xaxis.set_ticklabels(tick_labels, rotation=45)
    axs[0, 0].set_xlabel('sequence length')
    axs[0, 0].set_ylabel('MAE')
    
    if metric == 'mae':
        axs[0, 0].set_ylim(0, 12)
    elif metric == 'sae':
        axs[0, 0].set_ylim(0, 1.2)
    
    # Microwave
    if metric == 'mae':
        cnn_mae = np.array([4.295420, 4.195645, 4.181283, 4.196355, 4.025705, 3.999147])
        rnn_mae = np.array([3.998947, 4.570398, 5.201335, 4.024450, 4.269150, 4.513125])
        wavenet_mae = np.array([4.331045, 4.266902, 4.252000, 4.272780, 4.077312, 4.066100, 4.244537, 3.958032])
    elif metric == 'sae':
        cnn_mae = np.array([0.955187, 0.909517, 0.898725, 0.923340, 0.769628, 0.825410])
        rnn_mae = np.array([0.728400, 1.034825, 0.348050, 0.796475, 0.917050, 1.019950])
        wavenet_mae = np.array([0.968350, 0.938490, 0.924618, 0.930350, 0.835078, 0.868263, 0.917895, 0.775090])
    
    tick_labels = ['15', '31', '63', '127', '255', '511', '1023', '2047']

    axs[0, 1].scatter(np.arange(len(cnn_mae)), cnn_mae, color='g')
    axs[0, 1].scatter(np.arange(len(rnn_mae)), rnn_mae, color='b')
    axs[0, 1].scatter(np.arange(len(wavenet_mae)), wavenet_mae, color='r')
    
    line_cnn, = axs[0, 1].plot(cnn_mae, color='g')
    line_rnn, = axs[0, 1].plot(rnn_mae, color='b')
    line_wavenet, = axs[0, 1].plot(wavenet_mae, color='r')
    axs[0, 1].legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs[0, 1].set_title('Microwave')
    axs[0, 1].xaxis.set_ticks(np.arange(len(tick_labels)))
    axs[0, 1].xaxis.set_ticklabels(tick_labels, rotation=45)
    axs[0, 1].set_xlabel('sequence length')
    axs[0, 1].set_ylabel('MAE')
    
    if metric == 'mae':
        axs[0, 1].set_ylim(0, 12)
    elif metric == 'sae':
        axs[0, 1].set_ylim(0, 1.2)
    
    # Dish washer
    if metric == 'mae':
        cnn_mae = np.array([20.798375, 20.908775, 23.818800, 20.895300, 20.826375, 20.912575])
        rnn_mae = np.array([28.019575, 31.822325, 20.836700, 19.420140, 17.323795, 15.789973])
        wavenet_mae = np.array([20.794075, 20.841900, 20.960225, 17.888463, 14.203318, 17.264717, 20.817150, 10.972295])
    elif metric == 'sae':
        cnn_mae = np.array([1.000748, 0.992037, 0.582330, 0.992897, 0.997295, 1.007217])
        rnn_mae = np.array([0.865485, 1.085665, 0.624445, 0.554460, 0.498328, 0.395370])
        wavenet_mae = np.array([0.999353, 0.996302, 0.988758, 0.464230, 0.280605, 0.360525, 0.997880, 0.198522])
    
    tick_labels = ['15', '31', '63', '127', '255', '511', '1023', '2047']

    axs[1, 0].scatter(np.arange(len(cnn_mae)), cnn_mae, color='g')
    axs[1, 0].scatter(np.arange(len(rnn_mae)), rnn_mae, color='b')
    axs[1, 0].scatter(np.arange(len(wavenet_mae)), wavenet_mae, color='r')
    
    line_cnn, = axs[1, 0].plot(cnn_mae, color='g')
    line_rnn, = axs[1, 0].plot(rnn_mae, color='b')
    line_wavenet, = axs[1, 0].plot(wavenet_mae, color='r')
    axs[1, 0].legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs[1, 0].set_title('Dish washer')
    axs[1, 0].xaxis.set_ticks(np.arange(len(tick_labels)))
    axs[1, 0].xaxis.set_ticklabels(tick_labels, rotation=45)
    axs[1, 0].set_xlabel('sequence length')
    axs[1, 0].set_ylabel('MAE')
    
    if metric == 'mae':
        axs[1, 0].set_ylim(0, 35)
    elif metric == 'sae':
        axs[1, 0].set_ylim(0, 1.2)
    
    # Washing machine
    if metric == 'mae':
        cnn_mae = np.array([5.484893, 6.007343, 5.935147, 6.161442, 5.563615, 6.397975])
        rnn_mae = np.array([5.341025, 5.494175, 6.097975, 7.743600, 6.752550, 4.578900])
        wavenet_mae = np.array([6.085545, 5.333560, 4.945005, 4.718562, 3.879237, 3.657655, 3.653910, 3.666620])
    elif metric == 'sae':
        cnn_mae = np.array([0.512212, 0.533647, 0.700490, 0.656965, 0.665617, 0.689018])
        cnn_mae = np.array([0.463500, 0.454325, 0.615450, 0.720350, 0.803875, 0.539125])
        wavenet_mae = np.array([0.677443, 0.643770, 0.410032, 0.489862, 0.374965, 0.310862, 0.336580, 0.341470])
    
    tick_labels = ['15', '31', '63', '127', '255', '511', '1023', '2047']

    axs[1, 1].scatter(np.arange(len(cnn_mae)), cnn_mae, color='g')
    axs[1, 1].scatter(np.arange(len(rnn_mae)), rnn_mae, color='b')
    axs[1, 1].scatter(np.arange(len(wavenet_mae)), wavenet_mae, color='r')
    
    line_cnn, = axs[1, 1].plot(cnn_mae, color='g')
    line_rnn, = axs[1, 1].plot(rnn_mae, color='b')
    line_wavenet, = axs[1, 1].plot(wavenet_mae, color='r')
    axs[1, 1].legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs[1, 1].set_title('Washing machine')
    axs[1, 1].xaxis.set_ticks(np.arange(len(tick_labels)))
    axs[1, 1].xaxis.set_ticklabels(tick_labels, rotation=45)
    axs[1, 1].set_xlabel('sequence length')
    axs[1, 1].set_ylabel('MAE')
    
    if metric == 'mae':
        axs[1, 1].set_ylim(0, 12)
    elif metric == 'sae':
        axs[1, 1].set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.show()


def computation_time():
    
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    
    # Kettle
    cnn_time = np.array([4, 3, 5, 32, 125, 456, 1765, 6900])
    rnn_time = np.array([25, 30, 26, 40, 70, 180, 300, 450])
    wavenet_time = np.array([16, 22, 31, 46, 63, 30, 70, 143])
    
    tick_labels = ['15', '31', '63', '127', '255', '511', '1023', '2047']

    axs.scatter(np.arange(len(cnn_time)), cnn_time, color='g')
    axs.scatter(np.arange(len(rnn_time)), rnn_time, color='b')
    axs.scatter(np.arange(len(wavenet_time)), wavenet_time, color='r')
    
    line_cnn, = axs.plot(cnn_time, color='g')
    line_rnn, = axs.plot(rnn_time, color='b')
    line_wavenet, = axs.plot(wavenet_time, color='r')
    axs.legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs.set_title('Computation complexity')
    axs.xaxis.set_ticks(np.arange(len(tick_labels)))
    axs.xaxis.set_ticklabels(tick_labels, rotation=45)
    axs.set_xlabel('sequence length')
    axs.set_ylabel('Time (ms) / iteration')
    
    plt.tight_layout()
    plt.show()


def width():

    metric = 'mae'
    
    fig, axs = plt.subplots(2,2, figsize=(7, 7))
    
    # Kettle
    if metric == 'mae':
        cnn_mae = np.array([10.408714, 10.366883, 6.849052])
        rnn_mae = np.array([10.548038, 10.172180, 4.630170])
        wavenet_mae = np.array([10.214536, 5.179978, 4.672648])
    
    tick_labels = ['1', '10', '100']

    axs[0, 0].scatter(np.arange(len(cnn_mae)), cnn_mae, color='g')
    axs[0, 0].scatter(np.arange(len(rnn_mae)), rnn_mae, color='b')
    axs[0, 0].scatter(np.arange(len(wavenet_mae)), wavenet_mae, color='r')
    
    line_cnn, = axs[0, 0].plot(cnn_mae, color='g')
    line_rnn, = axs[0, 0].plot(rnn_mae, color='b')
    line_wavenet, = axs[0, 0].plot(wavenet_mae, color='r')
    axs[0, 0].legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs[0, 0].set_title('Kettle')
    axs[0, 0].xaxis.set_ticks(np.arange(len(tick_labels)))
    axs[0, 0].xaxis.set_ticklabels(tick_labels, rotation=45)
    axs[0, 0].set_xlabel('sequence length')
    axs[0, 0].set_ylabel('MAE')
    
    if metric == 'mae':
        axs[0, 0].set_ylim(0, 12)
    
    # Microwave
    if metric == 'mae':
        cnn_mae = np.array([4.115545, 4.069686, 3.999152])
        rnn_mae = np.array([16.056990, 27.650613, 3.998954])
        wavenet_mae = np.array([4.409568, 4.359637, 3.958039])
    
    tick_labels = ['1', '10', '100']

    axs[0, 1].scatter(np.arange(len(cnn_mae)), cnn_mae, color='g')
    axs[0, 1].scatter(np.arange(len(rnn_mae)), rnn_mae, color='b')
    axs[0, 1].scatter(np.arange(len(wavenet_mae)), wavenet_mae, color='r')
    
    line_cnn, = axs[0, 1].plot(cnn_mae, color='g')
    line_rnn, = axs[0, 1].plot(rnn_mae, color='b')
    line_wavenet, = axs[0, 1].plot(wavenet_mae, color='r')
    axs[0, 1].legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs[0, 1].set_title('Microwave')
    axs[0, 1].xaxis.set_ticks(np.arange(len(tick_labels)))
    axs[0, 1].xaxis.set_ticklabels(tick_labels, rotation=45)
    axs[0, 1].set_xlabel('sequence length')
    axs[0, 1].set_ylabel('MAE')
    
    if metric == 'mae':
        axs[0, 1].set_ylim(0, 40)
    
    # Dish washer
    if metric == 'mae':
        cnn_mae = np.array([20.895197, 20.852929, 20.798424])
        rnn_mae = np.array([21.700007, 19.014683, 15.790031])
        wavenet_mae = np.array([20.810505, 20.792687, 10.972319])
    
    tick_labels = ['1', '10', '100']

    axs[1, 0].scatter(np.arange(len(cnn_mae)), cnn_mae, color='g')
    axs[1, 0].scatter(np.arange(len(rnn_mae)), rnn_mae, color='b')
    axs[1, 0].scatter(np.arange(len(wavenet_mae)), wavenet_mae, color='r')
    
    line_cnn, = axs[1, 0].plot(cnn_mae, color='g')
    line_rnn, = axs[1, 0].plot(rnn_mae, color='b')
    line_wavenet, = axs[1, 0].plot(wavenet_mae, color='r')
    axs[1, 0].legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs[1, 0].set_title('Dish washer')
    axs[1, 0].xaxis.set_ticks(np.arange(len(tick_labels)))
    axs[1, 0].xaxis.set_ticklabels(tick_labels, rotation=45)
    axs[1, 0].set_xlabel('sequence length')
    axs[1, 0].set_ylabel('MAE')
    
    if metric == 'mae':
        axs[1, 0].set_ylim(0, 35)
    
    # Washing machine
    if metric == 'mae':
        cnn_mae = np.array([6.068237, 6.047284, 5.484898])
        rnn_mae = np.array([6.339066, 5.465815, 4.578951])
        wavenet_mae = np.array([6.232575, 4.141722, 3.653914])
        
    
    tick_labels = ['1', '10', '100']

    axs[1, 1].scatter(np.arange(len(cnn_mae)), cnn_mae, color='g')
    axs[1, 1].scatter(np.arange(len(rnn_mae)), rnn_mae, color='b')
    axs[1, 1].scatter(np.arange(len(wavenet_mae)), wavenet_mae, color='r')
    
    line_cnn, = axs[1, 1].plot(cnn_mae, color='g')
    line_rnn, = axs[1, 1].plot(rnn_mae, color='b')
    line_wavenet, = axs[1, 1].plot(wavenet_mae, color='r')
    axs[1, 1].legend(handles=[line_cnn, line_rnn, line_wavenet], labels=['CNN', 'RNN', 'WaveNet'])
    
    axs[1, 1].set_title('Washing machine')
    axs[1, 1].xaxis.set_ticks(np.arange(len(tick_labels)))
    axs[1, 1].xaxis.set_ticklabels(tick_labels, rotation=45)
    axs[1, 1].set_xlabel('sequence length')
    axs[1, 1].set_ylabel('MAE')
    
    if metric == 'mae':
        axs[1, 1].set_ylim(0, 12)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('mae')
    parser_b = subparsers.add_parser('computation_time')
    parser_c = subparsers.add_parser('width')
    
    args = parser.parse_args()
    
    if args.mode == 'mae':
        mae()
        
    elif args.mode == 'computation_time':
        computation_time()
        
    elif args.mode == 'width':
        width()
        
    else:
        raise Exception('Incorrect argument!')