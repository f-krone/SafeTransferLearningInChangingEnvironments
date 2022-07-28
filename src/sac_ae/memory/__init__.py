from .replay_drq import *
from .replay_cost import *

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

def make_replay_buffer(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, obs_shape, device, image_size, image_pad, robot, save_cost):
    max_size_per_worker = max_size // max(1, num_workers)

    dataset_class, buffer_class = (ReplayBufferDatasetCost, ReplayBufferCost) if save_cost else (ReplayBufferDataset, ReplayBuffer)

    iterable = dataset_class(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    buffer = buffer_class(iter(loader), obs_shape, device, image_size, image_pad, robot)
    
    return buffer