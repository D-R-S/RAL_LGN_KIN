# lgn_hand_ik

git clone into this repo!

run chmod u+x on all .sh scripts

run 

> install_lgn_bench.sh // arm.sh

then 

> run_lgn_bench.sh // arm.sh

# sim

navigate to sim subfolder after git clone

> chmake .. -DBUILD_ROS=OFF -DBUILD_DEMO=OFF -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=ON
then 
> make -j twin_sim

run via

> ./twin_sim

see various runtime flags --n N, --damping D, --diff a b, --chaos ... 
