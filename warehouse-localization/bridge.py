"""Go2 observations -> isolated TF/scan and validated AMCL pose. No motion publisher."""
import json,math,os,queue,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2,LaserScan
from geometry_msgs.msg import TransformStamped,PoseWithCovarianceStamped
from std_msgs.msg import String
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster
from sensor_msgs_py import point_cloud2
from quality import fresh,yaw,project_cloud,pose_valid,status

def stamp(header):return header.stamp.sec+header.stamp.nanosec/1e9

class Bridge(Node):
    def __init__(self):
        super().__init__('sensor_bridge',namespace='wendy_localization')
        self.mode=os.getenv('LOCALIZATION_MODE','mapping')
        self.stamps={};self.pose=None;self.rejections={};self.initial=queue.Queue();self.pending=None
        self.measurements=queue.Queue(maxsize=4)
        self.lock=threading.RLock()
        self.tf=TransformBroadcaster(self)
        self.scan_pub=self.create_publisher(LaserScan,'scan',qos_profile_sensor_data)
        self.odom_pub=self.create_publisher(Odometry,'odom',qos_profile_sensor_data)
        self.pose_pub=self.create_publisher(PoseWithCovarianceStamped,'pose',10)
        self.status_pub=self.create_publisher(String,'status',10)
        self.initial_pub=self.create_publisher(PoseWithCovarianceStamped,'initialpose',10)
        self.nomotion=self.create_client(Empty,'request_nomotion_update')
        self.create_subscription(Odometry,os.getenv('ODOM_TOPIC','/utlidar/robot_odom'),self.odom,qos_profile_sensor_data)
        self.create_subscription(PointCloud2,os.getenv('CLOUD_TOPIC','/utlidar/cloud_base'),self.cloud,qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped,'amcl_pose',self.amcl,10)
        self.create_timer(.1,self.tick)
        self.snapshot=status(0,{},None,self.mode)

    def now(self):return self.get_clock().now().nanoseconds/1e9
    def reject(self,key,reason):
        with self.lock:self.stamps.pop(key,None);self.rejections[key]=str(reason)
    def odom(self,msg):
        try:
            if not fresh(stamp(msg.header),self.now()):raise ValueError('stale odometry capture or clock mismatch')
            if msg.header.frame_id!=os.getenv('INPUT_ODOM_FRAME','odom') or msg.child_frame_id!=os.getenv('INPUT_BASE_FRAME','base_link'):
                raise ValueError(f'uncommissioned odometry frames {msg.header.frame_id} -> {msg.child_frame_id}')
            p,q=msg.pose.pose.position,msg.pose.pose.orientation
            heading=yaw((q.x,q.y,q.z,q.w))
            if not all(math.isfinite(v) for v in (p.x,p.y,p.z)):raise ValueError('invalid position')
            msg.header.frame_id='wendy_odom';msg.child_frame_id='wendy_base'
            msg.pose.pose.position.z=0.;q.x=q.y=0.;q.z=math.sin(heading/2);q.w=math.cos(heading/2)
            tf=TransformStamped();tf.header=msg.header;tf.child_frame_id='wendy_base'
            tf.transform.translation.x=p.x;tf.transform.translation.y=p.y;tf.transform.rotation=q
            self.tf.sendTransform(tf);self.odom_pub.publish(msg)
            with self.lock:self.stamps['odom']=stamp(msg.header);self.rejections.pop('odom',None)
        except Exception as e:self.reject('odom',e)
    def cloud(self,msg):
        try:
            if not fresh(stamp(msg.header),self.now()):raise ValueError('stale LiDAR capture or clock mismatch')
            if msg.header.frame_id!=os.getenv('INPUT_CLOUD_FRAME','base_link'):
                raise ValueError(f'uncommissioned cloud frame {msg.header.frame_id}; require base-frame cloud')
            pts=point_cloud2.read_points_numpy(msg,field_names=('x','y','z'),skip_nans=True)
            ranges=project_cloud(pts,float(os.getenv('MIN_HEIGHT','.15')),float(os.getenv('MAX_HEIGHT','1.5')))
            scan=LaserScan();scan.header=msg.header;scan.header.frame_id='wendy_base'
            scan.angle_min=-math.pi;scan.angle_increment=2*math.pi/360;scan.angle_max=scan.angle_min+359*scan.angle_increment
            scan.range_min=.35;scan.range_max=12.;scan.scan_time=.1;scan.ranges=ranges.astype(float).tolist()
            self.scan_pub.publish(scan)
            with self.lock:self.stamps['scan']=stamp(msg.header);self.rejections.pop('scan',None)
        except Exception as e:self.reject('scan',e)
    def amcl(self,msg):
        try:
            if msg.header.frame_id!='wendy_map' or not fresh(stamp(msg.header),self.now(),2):raise ValueError('invalid map pose frame or age')
            if not pose_valid(msg.pose.covariance):raise ValueError('pose uncertainty exceeds limits')
            p,q=msg.pose.pose.position,msg.pose.pose.orientation
            heading=yaw((q.x,q.y,q.z,q.w))
            if not all(math.isfinite(v) for v in (p.x,p.y,heading)):raise ValueError('invalid pose')
            with self.lock:
                self.pose={'x':p.x,'y':p.y,'yaw':heading,'variance':[msg.pose.covariance[i] for i in (0,7,35)]}
                self.stamps['pose']=stamp(msg.header);self.rejections.pop('pose',None)
                valid=status(self.now(),self.stamps,self.pose,self.mode)['ready']
            if valid:self.pose_pub.publish(msg)
        except Exception as e:self.reject('pose',e)
    def tick(self):
        while not self.measurements.empty():
            values=self.measurements.get()
            odom=Odometry();odom.header.frame_id=os.getenv('INPUT_ODOM_FRAME','odom');odom.child_frame_id=os.getenv('INPUT_BASE_FRAME','base_link')
            capture=values['capture_time'];odom.header.stamp.sec=int(capture);odom.header.stamp.nanosec=int((capture-int(capture))*1e9)
            odom.pose.pose.position.x=values['odom'][0];odom.pose.pose.position.y=values['odom'][1]
            odom.pose.pose.orientation.z=math.sin(values['odom'][2]/2);odom.pose.pose.orientation.w=math.cos(values['odom'][2]/2)
            self.odom(odom)
            points=[]
            for i,r in enumerate(values['ranges']):
                if r is not None and math.isfinite(r) and .35<=r<=12:
                    a=-math.pi+i*2*math.pi/360;points.append([r*math.cos(a),r*math.sin(a),.8])
            header=odom.header;header.frame_id=os.getenv('INPUT_CLOUD_FRAME','base_link')
            self.cloud(point_cloud2.create_cloud_xyz32(header,points))
        with self.lock:
            self.snapshot=status(self.now(),self.stamps,self.pose,self.mode)
            self.snapshot['rejections']=dict(self.rejections)
            self.snapshot['simulated_input']=os.getenv('ALLOW_SIMULATED_INPUT')=='true'
            self.status_pub.publish(String(data=json.dumps(self.snapshot)))
        if self.pending is None or self.pending.done():
            if self.mode=='localization' and self.nomotion.service_is_ready():self.pending=self.nomotion.call_async(Empty.Request())
        if not self.initial.empty():
            values=self.initial.get();msg=PoseWithCovarianceStamped();msg.header.stamp=self.get_clock().now().to_msg();msg.header.frame_id='wendy_map'
            msg.pose.pose.position.x=values['x'];msg.pose.pose.position.y=values['y']
            msg.pose.pose.orientation.z=math.sin(values['yaw']/2);msg.pose.pose.orientation.w=math.cos(values['yaw']/2)
            for i,value in zip((0,7,35),values['variance']):msg.pose.covariance[i]=value
            self.initial_pub.publish(msg)

def main():
    rclpy.init();node=Bridge()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with node.lock:payload=json.dumps(node.snapshot).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_POST(self):
            if self.path=='/measurements' and os.getenv('ALLOW_SIMULATED_INPUT')=='true':
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=20000:raise ValueError('request too large')
                    values=json.loads(self.rfile.read(length))
                    assert len(values['odom'])==3 and all(math.isfinite(v) for v in values['odom'])
                    assert len(values['ranges'])==360 and math.isfinite(values['capture_time'])
                    node.measurements.put_nowait(values);self.send_response(202);self.end_headers()
                except Exception:self.send_error(400,'invalid simulated measurement or full queue')
                return
            if self.path!='/initialpose' or node.mode!='localization':self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2048:raise ValueError('invalid request size')
                values=json.loads(self.rfile.read(length))
                assert all(math.isfinite(values[k]) for k in ('x','y','yaw'))
                assert len(values['variance'])==3 and all(.001<=v<=4 for v in values['variance'])
                node.initial.put(values);self.send_response(202);self.end_headers()
            except Exception:self.send_error(400,'Need finite x, y, yaw and three positive variances')
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('0.0.0.0',int(os.getenv('PORT','8790'))),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:rclpy.spin(node)
    finally:server.shutdown();server.server_close();node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
