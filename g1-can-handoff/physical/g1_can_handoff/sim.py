"""Physical contact simulation. No object attachment or scripted object motion."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
ARM = [f"left_{name}_joint" for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
FINGERS = [f"left_hand_{name}_joint" for name in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]

def scene(height=.75, x=.35, y=.18, mass=.37, friction=1.0, decorate=None):
    root = ET.parse(ROOT / "assets/unitree/g1.xml").getroot()
    root.find("compiler").set("meshdir", str(ROOT / "assets/unitree/meshes"))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    ET.SubElement(root, "visual").append(ET.Element("global", offwidth="640", offheight="480"))
    world = root.find("worldbody")
    pelvis = world.find("body")
    pelvis.remove(pelvis.find("joint"))  # Supported base, not a balance test.
    # Convex mesh hulls intersect at the shoulder mount. These neighboring
    # structural links are excluded, while arm/body and arm/table contacts remain.
    contact=ET.SubElement(root,"contact")
    for side in ("left","right"):
        for axis in ("roll","yaw"):
            ET.SubElement(contact,"exclude",body1="torso_link",body2=f"{side}_shoulder_{axis}_link")
    for joint in root.findall(".//joint"):
        if joint.get("name"):
            joint.set("damping", "0.08" if "hand" in joint.get("name") else "1")
            joint.set("armature", "0.002" if "hand" in joint.get("name") else "0.01")
    for geom in root.findall(".//geom"):
        if geom.get("contype") != "0":
            geom.set("friction", "1 0.01 0.001")
            geom.set("condim", "4")
    palm = root.find(".//body[@name='left_wrist_yaw_link']")
    ET.SubElement(palm, "site", name="grasp", pos="0.14 -0.045 0", size="0.006", rgba="0 1 0 0.5")
    if root.find(".//geom[@name='floor']") is None:
        ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 .1", rgba=".16 .19 .23 1")
    ET.SubElement(world, "geom", name="table", type="box", pos=f".48 .18 {height-.025}", size=".25 .32 .025", rgba=".55 .40 .24 1")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse=".9 .9 .9")
    cam_pos = np.array([.07, -.10, 1.40])
    z_axis = cam_pos - np.array([.40, .18, .80]); z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0, 0, 1], z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    ET.SubElement(world, "camera", name="observe", pos=" ".join(map(str,cam_pos)), xyaxes=" ".join(map(str,np.r_[x_axis,y_axis])), fovy="65")
    # Approximate head-mounted viewpoint, fixed to the upper body and pitched
    # down 45 degrees toward the workspace. This is not a calibrated G1 sensor.
    torso = root.find(".//body[@name='torso_link']")
    ET.SubElement(torso, "camera", name="head", pos=".084 0 .393",
                  xyaxes="0 -1 0 .70710678 0 .70710678", fovy="75")
    can = ET.SubElement(world, "body", name="can", pos=f"{x} {y} {height+.061}")
    ET.SubElement(can, "freejoint", name="can_free")
    ET.SubElement(can, "geom", name="can_shell", type="cylinder", size=".033 .061", mass=str(mass), rgba=".85 .025 .035 1", friction=f"{friction} .01 .001", condim="4", margin=".01", gap=".01")
    ET.SubElement(can, "geom", name="can_top", type="cylinder", pos="0 0 .061", size=".032 .0005", rgba=".8 .8 .8 1", contype="0", conaffinity="0", mass="0")
    if decorate is not None:
        decorate(root)
    return Plant(mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")))

class Plant:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ik_data = mujoco.MjData(model)
        self.joints = np.array(model.actuator_trnid[:, 0])
        self.qadr = model.jnt_qposadr[self.joints]
        self.dadr = model.jnt_dofadr[self.joints]
        self.targets = self.data.qpos[self.qadr].copy()
        self.arm = np.array([model.actuator(n).id for n in ARM])
        self.fingers = np.array([model.actuator(n).id for n in FINGERS])
        self.data.qpos[self.qadr[self.arm]] = [-1.3,.8,0,1.2,0,0,0]
        self.targets = self.data.qpos[self.qadr].copy()
        self.kp = np.array([12 if "hand" in model.actuator(i).name else 100 for i in range(model.nu)])
        self.kd = np.array([.2 if "hand" in model.actuator(i).name else 4 for i in range(model.nu)])
        self.torque = np.max(np.abs(model.jnt_actfrcrange[self.joints]), axis=1)
        self.site = model.site("grasp").id
        self.can_id = model.body("can").id
        self.frame_callback = None
        self.contact_phase = "pregrasp"
        mujoco.mj_forward(model, self.data)

    def step(self, seconds):
        for _ in range(round(seconds / self.model.opt.timestep)):
            m, d = self.model, self.data
            effort = self.kp*(self.targets-d.qpos[self.qadr]) - self.kd*d.qvel[self.dadr] + d.qfrc_bias[self.dadr]
            d.ctrl[:] = np.clip(effort, -self.torque, self.torque)
            mujoco.mj_step(m, d)
            if self.frame_callback: self.frame_callback(self)

    def solve(self, position, rotation=None):
        m,d = self.model,self.ik_data
        d.qpos[:] = self.data.qpos
        adr = self.qadr[self.arm]
        lo,hi = m.jnt_range[self.joints[self.arm]].T
        lo,hi = lo+.025,hi-.025
        rotation = np.eye(3) if rotation is None else rotation
        def error(q):
            d.qpos[adr] = q
            mujoco.mj_forward(m,d)
            return np.r_[d.site_xpos[self.site]-position, .15*(d.site_xmat[self.site].reshape(3,3)-rotation).ravel()]
        seeds = [np.clip(self.data.qpos[adr],lo,hi), np.clip([-1.2,.2,0,1.2,0,0,0],lo,hi), np.clip([-.5,.2,1.3,1.4,.7,-.9,-.5],lo,hi)]
        candidates = [least_squares(error,s,bounds=(lo,hi),max_nfev=250) for s in seeds]
        valid=[]
        for r in candidates:
            error(r.x)
            if np.linalg.norm(r.fun[:3]) < .008 and np.linalg.norm(r.fun[3:]) < .02 and not self.pose_collision(d): valid.append(r)
        best = min(valid,key=lambda r: np.linalg.norm(r.x-self.data.qpos[adr])) if valid else min(candidates,key=lambda r: np.linalg.norm(r.fun))
        error(best.x)
        if np.linalg.norm(d.site_xpos[self.site]-position) > .012 or np.linalg.norm(d.site_xmat[self.site].reshape(3,3)-rotation) > .18:
            raise ValueError("Target cannot be reached with the required grasp orientation")
        collision=self.collision_details(d)
        if collision: raise ValueError(f"Target pose collision: {collision}")
        return best.x

    def set_contact_phase(self, phase):
        if phase not in ('pregrasp','staging','approach','align','reach','close','lift','hold','offer'):
            raise ValueError('Unknown collision phase')
        self.contact_phase=phase

    def collision_details(self,data):
        m=self.model
        def detail(contact,names):
            return dict(phase=self.contact_phase,bodies=names,
                        geoms=[m.geom(g).name for g in (contact.geom1,contact.geom2)],
                        distance_m=float(contact.dist))
        for contact in data.contact:
            names=[m.body(m.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
            geoms=[m.geom(g).name for g in (contact.geom1,contact.geom2)]
            if 'can' in names:
                other=names[1] if names[0]=='can' else names[0]
                # Table support is expected. Only the designated left fingers
                # may touch the can, and only in the intentional grasp phases.
                if other=='world':continue
                intentional=(self.contact_phase in ('reach','close','lift','hold','offer') and
                             any(other.startswith('left_hand_'+finger) for finger in ('thumb','index','middle')))
                if intentional:continue
                threshold=.01 if self.contact_phase in ('pregrasp','staging','approach','align') else 0.
                if contact.dist<threshold:return detail(contact,names)
                continue
            clearance=getattr(self,'table_clearance_m',0.)
            if ('table' in geoms and clearance>0 and
                    any(n.startswith('left_') for n in names) and contact.dist<clearance):
                return dict(**detail(contact,names),required_clearance_m=clearance)
            if contact.dist>=-.001:continue
            if all("left_hand" in n or n=="left_wrist_yaw_link" for n in names):continue
            if any(n.startswith("left_") for n in names) and any(n in ("world","torso_link","pelvis") or n.startswith("right_") for n in names):return detail(contact,names)
        return None

    def pose_collision(self,data):
        return self.collision_details(data) is not None

    def move(self, position, duration=2., rotation=None):
        goal = self.solve(np.array(position),rotation)
        start = self.targets[self.arm].copy()
        for t in np.linspace(0,1,round(duration/.02)):
            alpha = t*t*(3-2*t)
            self.targets[self.arm] = start+alpha*(goal-start)
            self.step(.02)
        self.step(.25)
        if np.linalg.norm(self.data.site_xpos[self.site]-position)>.025:
            raise ValueError("Arm failed to track the target; possible collision or torque limit")

    def hand(self, goal, duration=1.):
        start = self.targets[self.fingers].copy()
        for alpha in np.linspace(0,1,round(duration/.02)):
            self.targets[self.fingers] = start+alpha*(np.array(goal)-start)
            self.step(.02)

    def can_position(self):
        return self.data.xpos[self.can_id].copy()

    def contacts(self):
        groups = set()
        m,d = self.model,self.data
        for c in d.contact:
            if c.dist>0:continue
            b1,b2 = m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            other = b2 if b1==self.can_id else b1 if b2==self.can_id else -1
            if other < 0: continue
            name = m.body(other).name
            for finger in ("thumb","index","middle"):
                if f"left_hand_{finger}" in name: groups.add(finger)
        return sorted(groups)
