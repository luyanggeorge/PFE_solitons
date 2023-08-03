#$ -cwd -V
#$ -l h_rt=24:00:00
#$ -l h_vmem=5G
#$ -pe ib 40
#$ -m be

module swap openmpi mvapich2
module add apptainer

mpiexec -n 40 singularity exec --env 'PATH=/home/firedrake/firedrake/bin:$PATH' -B /run -B /nobackup -B ~/.cache:/home/firedrake/firedrake/.cache /home/home02/mmyl/firedrake_latest.sif python3 3D_tank_periodic.py
