# DataLoader
- __getitem__
	- need
		- self.im_refs
		- self.image_index, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships
	- input
		- ind
	- output 
		- img, target, index
```
target = BoxList(obj_boxes, (width, height), mode="xyxy")
target['labels']  (box label)
target['pred_labels']  (predicates)
target['relation_labels']  (bbox1, bbox2, label)
```

# data

imdb_1024.h5 
```
{
	'image_heights': <HDF5 dataset "image_heights": shape (43728,), type "<i4">, 
	'image_widths': <HDF5 dataset "image_widths": shape (43728,), type "<i4">, 
	'images': <HDF5 dataset "images": shape (43728, 3, 1024, 1024), type "|u1">, 
}
```
VG-SGG.h5 (boxes_1024: bbox value, img_to_first_box: images index, )
```
{
	'split': <HDF5 dataset "split": shape (43728,), type "<i4">
	'boxes_1024': <HDF5 dataset "boxes_1024": shape (414501, 4), type "<i4">, 
	'labels': <HDF5 dataset "labels": shape (414501, 1), type "<i8">, 
	'img_to_first_box': <HDF5 dataset "img_to_first_box": shape (43728,), type "<i4">, 
	'img_to_last_box': <HDF5 dataset "img_to_last_box": shape (43728,), type "<i4">, 
	'relationships': <HDF5 dataset "relationships": shape (250480, 2), type "<i4">, 
	'predicates': <HDF5 dataset "predicates": shape (250480, 1), type "<i8">, 
	'img_to_first_rel': <HDF5 dataset "img_to_first_rel": shape (43728,), type "<i4">, 
	'img_to_last_rel': <HDF5 dataset "img_to_last_rel": shape (43728,), type "<i4">, 
}
```

```
attribute
```
[{"image_id": 1, "attributes": [{"synsets": ["clock.n.01"], "h": 339, "object_id": 1058498, "names": ["clock"], "w": 79, "attributes": ["green", "tall"], "y": 91, "x": 421}, {"synsets": ["street.n.01"], "h": 262, "object_id": 5046, "names": ["street"], "w": 714, "attributes": ["sidewalk"], "y": 328, "x": 77}, {"synsets": ["shade.n.01"], "h": 192, "object_id": 5045, "names": ["shade"], "w": 274, "y": 338, "x": 119}, {"synsets": ["man.n.01"], "h": 262, "object_id": 1058529, "names": ["man"], "w": 60, "y": 249, "x": 238}, {"synsets": ["gym_shoe.n.01"], "h": 26, "object_id": 5048, "names": ["sneakers"], "w": 52, "attributes": ["grey"], "y": 489, "x": 243}, ]}, ]
```
predicates (250480, class label)
```
and
says
belonging to
over
parked on
growing on
standing on
made of
attached to
at
in
...
array([[49],
       [31],
       [20],
       ...,
       [31],
       [50],
       [31]])
```
relationships : 250480, bbox之間的relation
boxes_1024[relationships index[0]], boxes_1024[relationships index[1]]
```
array([[     7,     21],
       [     8,      2],
       [     7,     13],
       ...,
       [414500, 414496],
       [414496, 414498],
       [414497, 414496]], dtype=int32)
```