from isaacsim.util.debug_draw import _debug_draw
import carb

def debug_draw_obb(obb, color=(1,0,0)):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_lines(
        [carb.Float3(i) for i in obb[[0,1,2,2,0,4,5,5,7,6,7,6]]] , 
        [carb.Float3(i) for i in obb[[1,3,3,0,4,5,1,7,6,4,3,2]]] , 
        [carb.ColorRgba(color[0],color[1],color[2],1.0)]*12,
        [1]*12 )
    return draw

def debug_draw_points(points,size = 3, color=[1,0,0]):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_points(
        [carb.Float3(i) for i in points] , 
        [carb.ColorRgba(color[0],color[1],color[2],1.0)]*len(points),
        [size]*len(points) )
    return draw

def debug_draw_lines(arr, color=[1,0,0]):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_lines(
        [carb.Float3(i) for i in arr[:-1]] , 
        [carb.Float3(i) for i in arr[1:]] , 
        [carb.ColorRgba(color[0],color[1],color[2],1.0)]*(len(arr)-1),
        [1]*(len(arr)-1) )
    return draw

def debug_draw_clear():
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    draw.clear_lines()