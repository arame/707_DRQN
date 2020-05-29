from device import Device
import os
class Config:
    n_games = 1
    lr = 0.0001
    gamma = 0.99
    eps_min = 0.01
    eps_decay = 0.999
    epsilon = 1.0
    max_mem = 5000
    repeat = 4
    batch_size = 32
    replace_target_cnt = 1000
    env_name = 'PongNoFrameskip-v4'
    path = 'models'
    load_checkpoint = False
    algo = 'DQNAgent'
    clip_rewards = False
    no_ops = 0
    fire_first = False
    chkpt_dir='checkpoint'
    plots_dir='plots'
    device, device_type = Device.get()
    figure_file = plots_dir + '/' + algo + '_' + env_name + '_lr' + str(lr) +'_'  + str(n_games) + 'games.png'

    @staticmethod
    def create_directories():
        Config.create_directory(Config.path)
        Config.create_directory(Config.chkpt_dir)
        Config.create_directory(Config.plots_dir)

    @staticmethod
    def create_directory(dir):
        if os.path.isdir(dir) == False:
            os.mkdir(dir)

    @staticmethod
    def print_settings():
        print("\n"*10)
        print("*"*100)
        print("** settings **")
        print("number of games = ", Config.n_games)
        print("learning rate (alpha) = ", Config.lr)
        print("epsilon start = ", Config.epsilon)
        print("epsilon minimum = ", Config.eps_min)
        print("epsilon decay = ", Config.eps_decay)
        print("gamma = ", Config.gamma)
        print("batch_size = ", Config.batch_size)
        print("environment = ", Config.env_name)
        print("algorithm = ", Config.algo)
        print("output graph located in ", Config.figure_file)
        print("*"*100)

    @staticmethod
    def get_name(item):
        return Config.env_name+'_'+Config.algo+item

