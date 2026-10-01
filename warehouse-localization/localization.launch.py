"""SLAM mapping or AMCL localization. All output frames are isolated; no motors."""
import os,yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess,RegisterEventHandler,EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

def generate_launch_description():
    mode=os.getenv('LOCALIZATION_MODE','mapping');ns='wendy_localization'
    common={'use_sim_time':False}
    processes=[ExecuteProcess(cmd=['python3','/app/bridge.py'],output='screen')]
    if mode=='mapping':
        path=Path(get_package_share_directory('slam_toolbox'))/'config/mapper_params_online_async.yaml'
        params=yaml.safe_load(path.read_text())['slam_toolbox']['ros__parameters']
        params.update(common,odom_frame='wendy_odom',map_frame='wendy_map',base_frame='wendy_base',
            scan_topic='/wendy_localization/scan',map_name='/wendy_localization/map',use_lifecycle_manager=True,mode='mapping',
            enable_interactive_mode=False,minimum_travel_distance=.1,minimum_travel_heading=.1)
        processes.append(Node(package='slam_toolbox',executable='async_slam_toolbox_node',name='slam_toolbox',namespace=ns,
            parameters=[params],remappings=[('/map','/wendy_localization/map'),('/map_metadata','/wendy_localization/map_metadata')],output='screen'))
        names=['slam_toolbox']
    elif mode=='localization':
        mapfile=Path(os.getenv('MAP_FILE','/data/map.yaml'))
        if not mapfile.is_file():raise RuntimeError(f'Missing surveyed/saved map: {mapfile}')
        processes.append(Node(package='nav2_map_server',executable='map_server',name='map_server',namespace=ns,
            parameters=[dict(common,yaml_filename=str(mapfile),frame_id='wendy_map')],output='screen'))
        params=dict(common,base_frame_id='wendy_base',odom_frame_id='wendy_odom',global_frame_id='wendy_map',
            scan_topic='/wendy_localization/scan',tf_broadcast=True,robot_model_type='nav2_amcl::OmniMotionModel',
            laser_model_type='likelihood_field',laser_min_range=.35,laser_max_range=12.,max_beams=120,
            min_particles=500,max_particles=2000,alpha1=.1,alpha2=.1,alpha3=.1,alpha4=.1,alpha5=.1,
            update_min_d=.05,update_min_a=.05,set_initial_pose=False,transform_tolerance=.2)
        processes.append(Node(package='nav2_amcl',executable='amcl',name='amcl',namespace=ns,parameters=[params],output='screen'))
        names=['map_server','amcl']
    else:raise ValueError('LOCALIZATION_MODE must be mapping or localization')
    processes.append(Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='lifecycle_manager',namespace=ns,
        parameters=[dict(common,autostart=True,node_names=names,bond_timeout=4.,attempt_respawn_reconnection=False)],output='screen'))
    handlers=[RegisterEventHandler(OnProcessExit(target_action=p,on_exit=[EmitEvent(event=Shutdown(reason='Localization process exited'))])) for p in processes]
    return LaunchDescription(processes+handlers)
