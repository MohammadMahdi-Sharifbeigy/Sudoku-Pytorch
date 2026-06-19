from app.scripts_lib_pose import polygon_line_to_pose, order_tl_tr_br_bl


def test_order_canonical():
    # deliberately shuffled: BR, TL, BL, TR
    pts = [(0.9, 0.9), (0.1, 0.1), (0.1, 0.9), (0.9, 0.1)]
    tl, tr, br, bl = order_tl_tr_br_bl(pts)
    assert tl == (0.1, 0.1)
    assert tr == (0.9, 0.1)
    assert br == (0.9, 0.9)
    assert bl == (0.1, 0.9)


def test_polygon_line_to_pose_shape_and_bbox():
    line = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"  # TL,TR,BR,BL already
    out = polygon_line_to_pose(line)
    cols = out.split()
    assert len(cols) == 17          # class + cx cy w h + 4*(x y v)
    assert cols[0] == "0"
    cx, cy, w, h = map(float, cols[1:5])
    assert abs(cx - 0.5) < 1e-6 and abs(cy - 0.5) < 1e-6
    assert abs(w - 0.8) < 1e-6 and abs(h - 0.8) < 1e-6
    # every visibility flag == 2
    assert cols[7] == "2" and cols[10] == "2" and cols[13] == "2" and cols[16] == "2"
