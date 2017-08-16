import os
from argparse import ArgumentParser
import socket
import subprocess
from threading import Thread
from hostlist import expand_hostlist
env = os.environ

SLURM_JOB_NODELIST = str(env["SLURM_JOB_NODELIST"])
SLURM_NTASKS = int(env["SLURM_NTASKS"])
SLURM_CPUS_PER_TASK = env["SLURM_CPUS_PER_TASK"]
SLURM_GPUS = env["CUDA_VISIBLE_DEVICES"] if env["CUDA_VISIBLE_DEVICES"] else ''
# SLURM_JOB_NODELIST = 'gpu[1-2]'#str(env["SLURM_JOB_NODELIST"])
# SLURM_NTASKS = 3#int(env["SLURM_NTASKS"])
# SLURM_CPUS_PER_TASK = 1#env["SLURM_CPUS_PER_TASK"]
parser = ArgumentParser()


parser.add_argument("--lico_cpus_per_ps_task", help="the cpu numbers of per ps task")
parser.add_argument("--lico_cpus_per_worker_task", help="the cpu numbers of per worker task")
parser.add_argument("--lico_gpus_per_worker_task", help="the gpu numbers of per worker task")
parser.add_argument("--lico_node_number", help="node number")
parser.add_argument("--lico_mx_program", help="the path of mxnet program")
parser.add_argument("--lico_args", help="the args of task")

args = parser.parse_args()

CPUS_PER_PS_TASK = args.lico_cpus_per_ps_task
CPUS_PER_WORKER_TASK = args.lico_cpus_per_worker_task
GPUS_PER_WORKER_TASK = args.lico_gpus_per_worker_task
PS_NUM = WORKER_NUM = args.lico_node_number
PROGRAM = args.lico_mx_program
ARGS = args.lico_args

def lico_dl_run():
    hosts_list = format_hosts_list(SLURM_JOB_NODELIST)

    host = hosts_list[0]
    envs = {'DMLC_NUM_WORKER' : WORKER_NUM,
            'DMLC_NUM_SERVER' : PS_NUM}
    task_cmd = 'python ' + PROGRAM + ' ' + ARGS
    ps_root = PSTracker(host=host, env=envs)
    mx_env = ps_root.env
    export_env_root = []
    for k, v in mx_env.items():
        export_env_root.append('export ' + str(k) + '=' + str(v) + ';')
    export_mxnet_root_env = ' '.join(export_env_root)
    if GPUS_PER_WORKER_TASK:
        srun_mxnet_root = export_mxnet_root_env + " srun -N1 -n1 --cpu_bind=cores --cpus-per-task=1 --gres=gpu:0 --nodelist=%s -l %s "%(host, task_cmd)
    else:
        srun_mxnet_root = export_mxnet_root_env + " srun -N1 -n1 --cpu_bind=cores --cpus-per-task=1 --nodelist=%s -l %s "%(host, task_cmd)
    # srun_mxnet_root = " srun -N1 -n1 --cpus-per-task=1 --gres=gpu:0 --nodelist=%s -l %s"%(host, task_cmd)
    # os.system(srun_mxnet_root)
    thread_root = Thread(target=(lambda: os.system(srun_mxnet_root)), args=()) # start mxnet root
    # thread_root = Thread(target=(lambda: subprocess.call(srun_mxnet_root, env=mx_env, shell=True)), args=()) # start mxnet root
    thread_root.setDaemon(True)
    thread_root.start()

    worker_list = []
    for host in hosts_list:
        indx = str(hosts_list.index(host))

        # start mxnet ps
        mx_env['DMLC_ROLE'] = 'server'
        mx_env['DMLC_SERVER_ID'] = indx
        export_env_server = []
        for k, v in mx_env.items():
            export_env_server.append('export ' + str(k) + '=' + str(v) + ';')
        export_mxnet_server_env = ' '.join(export_env_server)
        if GPUS_PER_WORKER_TASK:
            srun_mxnet_server = export_mxnet_server_env + " srun -N1 -n1 --cpu_bind=cores --cpus-per-task=%s --gres=gpu:0 --nodelist=%s -l %s "% (CPUS_PER_PS_TASK, host, task_cmd)
        else:
            srun_mxnet_server = export_mxnet_server_env + " srun -N1 -n1 --cpu_bind=cores --cpus-per-task=%s --nodelist=%s -l %s "% (CPUS_PER_PS_TASK, host, task_cmd)
        # thread = Thread(target=(lambda: subprocess.call(srun_mxnet_server, env=mx_env, shell=True)), args=())
        # os.system(srun_mxnet_server)
        thread = Thread(target=(lambda: os.system(srun_mxnet_server)), args=())
        thread.setDaemon(True)
        thread.start()

       # start mxnet worker
        mx_env['DMLC_ROLE'] = 'worker'
        mx_env['DMLC_WORKER_ID'] = indx
        export_env_worker = []
        for k, v in mx_env.items():
            export_env_worker.append('export ' + str(k) + '=' + str(v) + ';')
        export_mxnet_worker_env = ' '.join(export_env_worker)
        # gpu_per_worker = '--gres=gpu:' + GPUS_PER_WORKER_TASK if int(GPUS_PER_WORKER_TASK) > 1 else '--gres=gpu:0'
        if SLURM_GPUS and SLURM_GPUS != 'NoDevFiles' and not '':
            task_cmd = 'python ' + PROGRAM + ' ' + ARGS + ' --gpus ' + SLURM_GPUS
        if GPUS_PER_WORKER_TASK:
            srun_mxnet_worker = export_mxnet_worker_env + " srun -N1 -n1 --cpu_bind=cores --cpus-per-task=%s --gres=gpu:%s --nodelist=%s -l %s "% (CPUS_PER_WORKER_TASK, str(GPUS_PER_WORKER_TASK), host, task_cmd)
        else:
            srun_mxnet_worker = export_mxnet_worker_env + " srun -N1 -n1 --cpu_bind=cores --cpus-per-task=%s --nodelist=%s -l %s "% (CPUS_PER_WORKER_TASK, host, task_cmd)
        # thread = Thread(target=(lambda: subprocess.call(srun_mxnet_worker, env=mx_env, shell=True)), args=())
        # os.system(srun_mxnet_worker)
        thread = Thread(target=(lambda: os.system(srun_mxnet_worker)), args=())
        worker_list.append(thread)
        thread.setDaemon(True)
        thread.start()
        mx_env.pop('DMLC_WORKER_ID')

    for w in worker_list:
        while w.isAlive():
            w.join(100)
    os.system('echo ---worker---done---!')

class PSTracker(object):
    """
    Tracker module for PS
    """
    def __init__(self, host, port=9091, port_end=9999, env=None):
        """
        Starts the PS scheduler
        """
        self.host = host
        hostIP = socket.gethostbyname(socket.getfqdn()) # IP
        family = socket.getaddrinfo(hostIP, None)[0][0] 
        sock = socket.socket(family, socket.SOCK_STREAM)
        for port in range(port, port_end):
            try:
                sock.bind(('', port))
                self.port = port
                sock.close()
                break
            except socket.error:
                continue

        env['DMLC_ROLE'] = 'scheduler'
        env['DMLC_PS_ROOT_URI'] = str(self.host)
        env['DMLC_PS_ROOT_PORT'] = str(self.port)
        self.env = env

 
def format_hosts_list(nodes):
    node_list = expand_hostlist(nodes)
    # gpu[1-2],a1,b2,c[3-4-5]
    # node_list = []
    # [node_list.append(n) if n.find('[') < 0 else __hosts_list_helper(n,node_list) for n in nodes.split(',')]
    return node_list     

# def __hosts_list_helper(node,list):
#     index = node.find('[')
#     [list.append(node[:index] + n)for n in node[index+1:-1].split('-')]
       

if __name__ == "__main__":
    lico_dl_run()
