import numpy as np

import quest.utils.metaworld_utils as mu
from quest.utils.draw_utils import show_point_cloud
from pyinstrument import Profiler
from tqdm import trange

boundaries = {'x_upper': .6,
              'x_lower': -.6,
              'y_upper': .95,
              'y_lower': .25,
              'z_upper': .3,
              'z_lower': 0,}

cam_names = ['topview', 'corner', 'corner2', 'corner3']

def axis_angle_to_rotation_matrix(axis, angle):
    """
    Convert an axis-angle representation to a rotation matrix.
    Parameters:
    axis (numpy.ndarray): A 3-element array representing the axis of rotation (must be a unit vector).
    angle (float): The angle of rotation in radians.
    Returns:
    numpy.ndarray: A 3x3 rotation matrix.
    """
    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    # Compute the skew-symmetric matrix K
    K = np.array([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
    ])
    # Compute the rotation matrix using Rodrigues' rotation formula
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def main():
    # env = mu.MetaWorldWrapper(
    #     env_name='basketball-v2',
    #     cam_names=cam_names,
    #     img_height=128,
    #     img_width=128,
    #     max_episode_length=500,
    #     env_kwargs=None,
    #     boundaries=boundaries,
    #     num_points=1024
    # )
    # env.reset()
    
    for env_name in mu._env_names:
        expert = mu.get_env_expert(env_name)
        # task_id = mu.get_index(env_name)
        env = mu.MetaWorldWrapper(
            env_name=env_name,
            cam_names=cam_names,
            img_height=128,
            img_width=128,
            max_episode_length=500,
            env_kwargs=None,
            boundaries=boundaries,
            num_points=128
        )
        init_obs = env.reset()
        # breakpoint()
        obs_gt = init_obs['obs_gt']
        done = False
        actions = []
        poss = [init_obs['agent_pos'][:3]]
        # profiler = Profiler()
        # profiler.start()
        while not done:
        # for _ in trange(500):
            action = expert.get_action(obs_gt)
            actions.append(action)
            next_obs, rew, terminated, truncated, info = env.step(action)
            obs_gt = next_obs['obs_gt']
            done = done or terminated or truncated
            poss.append(next_obs['agent_pos'][:3])
            # print(info['success'])
        # profiler.stop()
        # profiler.print()


        actions = np.array(actions)[:, :3] * env.env.action_scale
        # actions_abs = np.cumsum(actions, axis=0) + init_obs['agent_pos'][:3]
        # actions_abs = actions_abs[:20]
        # actions_abs_1 = np.concatenate((actions_abs, np.ones((20, 1))), axis=1)
        # actions_abs_transformed_v1 = (init_obs['hand_mat_inv'] @ actions_abs_1.T).T[:, :3]
        
        for _ in range(10):

            theta = np.random.random() * 2 * np.pi
            axis = (1, 0, 0)
            rot = axis_angle_to_rotation_matrix(axis, theta)

            scale = np.random.random(3) * 2.5 + .5
            scale = np.diag(scale)
            print(scale)
            # rot = np.eye(3)

            actions_transformed = (np.linalg.inv(init_obs['hand_rot_mat']) @ actions.T).T
            actions_transformed = actions_transformed @ rot.T
            breakpoint()
            actions_transformed = actions_transformed @ scale
            actions_transformed_abs = np.cumsum(actions_transformed, axis=0)
            actions_abs_transformed_v2 = actions_transformed_abs[:20]

            # poss = np.array(poss)
            # poss_1 = np.concatenate((poss, np.ones((len(poss), 1))), axis=1)
            # poss_transformed = (init_obs['hand_mat_inv'] @ poss_1.T).T[:, :3]

            color = np.array((252, 186, 3)) / 255
            color2 = np.array((252, 3, 232)) / 255
            color3 = np.array((3, 231, 252)) / 255
            # actions_abs_pointcloud_v1 = np.concatenate((actions_abs_transformed_v1, np.tile(color, (actions_abs_transformed_v1.shape[0], 1))), axis=1)
            actions_abs_pointcloud_v2 = np.concatenate((actions_abs_transformed_v2, np.tile(color2, (actions_abs_transformed_v2.shape[0], 1))), axis=1)
            # actions_abs_pointcloud_pos = np.concatenate((poss_transformed, np.tile(color3, (poss_transformed.shape[0], 1))), axis=1)
            # final_point_cloud = np.concatenate((init_obs['point_cloud'], actions_abs_pointcloud_v1, actions_abs_pointcloud_v2, actions_abs_pointcloud_pos))
            point_cloud_transformed = np.concatenate((init_obs['point_cloud'][:, :3] @ rot.T @ scale, init_obs['point_cloud'][:, 3:]), axis=1)
            # point_cloud_transformed = init_obs['point_cloud']
            final_point_cloud = np.concatenate((point_cloud_transformed, actions_abs_pointcloud_v2))


            # show_point_cloud(init_obs['point_cloud'])
            show_point_cloud(final_point_cloud)

        env.close()
        del env


if __name__ == '__main__':
    main()