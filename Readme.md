## 📚 CSCI 473 Project 2

To run:

```bash
roslaunch wall_following wall_following_v1.launch world:=triton_world.world
rosrun wall_following wallFollowing.py (mode)
```

`mode` can be `trained` or `training`.

`trained` mode uses a file called `table.npy`. I included it in the src/wall_following/src folder so you have to cd into that folder before running the command or copy it into your current directory. 
