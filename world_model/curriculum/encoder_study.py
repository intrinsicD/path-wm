"""Explicit entry points for the staged, authorized encoder investigation."""
import argparse


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['head','profile','prepare-dino','reference'])
    p.add_argument('--arm');p.add_argument('--seed',type=int,default=7107);p.add_argument('--development',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.command=='head':
        from .pose_diagnostic import train
        if a.arm not in ('coupled','independent'):p.error('head arm must be coupled or independent')
        train(a.arm,a.development,a.device)
    elif a.command=='profile':
        from .encoder_profile import profile
        profile(a.device)
    elif a.command=='prepare-dino':
        from .encoder_reference import prepare
        prepare(a.development,a.device)
    elif a.command=='reference':
        from .encoder_reference import train
        if a.arm not in ('custom','dino','native'):p.error('reference arm must be custom, dino, or native')
        train(a.arm,a.development,a.device)

if __name__=='__main__':main()
