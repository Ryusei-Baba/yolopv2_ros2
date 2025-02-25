from setuptools import setup
from glob import glob

package_name = 'yolopv2_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/data/weights', glob('data/weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ryusei Baba',
    maintainer_email='s20c1102kg@s.chibakoudai.jp',
    description='ROS2 node for YOLOPv2 object detection and BEV transformation.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolopv2_inference_node = yolopv2_ros2.yolopv2_inference_node:main',
            'bird_eye_view_node = yolopv2_ros2.bird_eye_view_node:main',
            'path_publisher_node = yolopv2_ros2.path_publisher_node:main',
            'follow_node = yolopv2_ros2.follow_node:main',
        ],
    },
)