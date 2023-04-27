import h5py

db = h5py.File(name="/Users/jongbeomkim/Desktop/workspace/SynthText/data/dset.h5", mode="r")
img = db["image"]["hiking_125.jpg"][:]
depth_map = db["depth"]["hiking_125.jpg"][:].T
seg_map = db["seg"]["hiking_125.jpg"][:]

renderer = RendererV3(data_dir="/Users/jongbeomkim/Desktop/workspace/SynthText/data")
area = db["seg"]["hiking_125.jpg"].attrs["area"]
label = db["seg"]["hiking_125.jpg"].attrs["label"]
renderer.render_text(rgb=img, depth=depth_map, seg=seg_map, area=area, label=label, ninstance=3, viz=True)




result = h5py.File("/Users/jongbeomkim/Desktop/workspace/SynthText/results/SynthText.h5", mode="r")
# result["data"].keys()
temp = result["data"]["hiking_125.jpg_0"][:]
show_image(temp)