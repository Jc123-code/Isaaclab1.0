from setuptools import setup, find_packages

setup(
    name='act-package',  # 改个名字避免冲突
    version='0.1.0',
    py_modules=['constants', 'policy', 'utils', 'ee_sim_env', 'sim_env', 'scripted_policy', 
                'imitate_episodes', 'record_sim_episodes', 'visualize_episodes'],
    packages=['detr', 'detr.models', 'detr.util'],
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
    ],
)
