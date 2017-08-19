import torch

# Bounding boxes and points.
bboxes = [[0, 0, 30, 30], [50, 50, 100, 100]]
bboxes = torch.FloatTensor(bboxes)
points = [[99, 50], [101, 0], [30, 30]]
points = torch.FloatTensor(points)

# Keep a reference to the original strided `points` tensor.
old_points = points

# Permutate all points for every single bounding box.
points = points.unsqueeze(1)
points = points.repeat(1, len(bboxes), 1)

# Create the conditions necessary to determine if a point is within a bounding box.
# x >= left, x <= right, y >= top, y <= bottom
c1 = points[:, :, 0] <= bboxes[:, 2]
c2 = points[:, :, 0] >= bboxes[:, 0]
c3 = points[:, :, 1] <= bboxes[:, 3]
c4 = points[:, :, 1] >= bboxes[:, 1]

# Add all of the conditions together. If all conditions are met, sum is 4.
# Afterwards, get all point indices that meet the condition (a.k.a. all non-zero mask-summed values)
mask = c1 + c2 + c3 + c4
mask = torch.nonzero((mask == 4).sum(dim=-1)).squeeze()

# Select all points that meet the condition.
print(old_points.index_select(dim=0, index=mask))
