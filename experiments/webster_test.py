# System Configuration check
import sys,os
sys.path.append(r'C:\Users\win11\OneDrive\paper\Accumulation\sumo_learning_new\sumo-rl')

import traci
class Simulation(object):
    def __init__(self, net_config, route_config, GUI=False):
        self.net_config = net_config
        self.route_config = route_config
        self.launch_env_flag = False
        self.GUI = GUI
    def launchEnv(self):
        """开始模拟(通过traci来获得其中数据)
        """
        if self.GUI:
            sumo_gui = 'sumo-gui'
        else:
            sumo_gui = 'sumo'
        traci.start([
            sumo_gui,
            "-n", self.net_config,
            "-r", self.route_config,
            "--seed", "2",
            "--duration-log.statistics",
            '--tripinfo-output.write-unfinished', 
            "--statistics-output","webster.xml"])
        self.launch_env_flag = True
    def close(self):
        """关闭实验环境
        """
        traci.close()
        self.launch_env_flag = False
        sys.stdout.flush()
    def reset(self):
        """关闭当前环境, 并开启一个新的环境
        """
        self.close()
        self.launchEnv()
    def step(self):
        assert self.launch_env_flag
        time = 0
        while time < 4e5 :
            print(f'current time = {time}\n')
            traci.simulationStep()
            time = traci.simulation.getCurrentTime()
    def actuated_control(self,actuated_phase_next,min_green,max_green,unit_green,current_green_time,current_time,phase_now):
        '''
        <tlLogic id="J1" type="static" programID="0" offset="0">
        0<phase duration="15" state="GGrGrrGGrGrr"/>
        <phase duration="3"  state="GyrGrrGyrGrr"/>
        <phase duration="1"  state="rrrrrrrrrrrr"/>
        3<phase duration="15" state="rrGrrrrrGrrr"/>
        <phase duration="3"  state="rryGrrGryGrr"/>
        <phase duration="1"  state="rrrrrrrrrrrr"/>
        6<phase duration="15" state="GrrGGrGrrGGr"/>
        <phase duration="3"  state="GrrGyrGrrGyr"/>
        <phase duration="1"  state="rrrrrrrrrrrr"/>
        9<phase duration="15" state="rrrrrGrrrrrG"/>
        <phase duration="3"  state="GrrGryGrrGry"/>
        <phase duration="1"  state="rrrrrrrrrrrr"/>
    </tlLogic>
        '''
        #相位序号：0-3-6-9-12(0)
        #维护下一个相位是什么
        if actuated_phase_next == phase_now:
            phase_next = phase_now
        else:
            phase_next = phase_now + 3
        if phase_next > 10:
            phase_next -= 12
        #主分支维护相位持续时间
        if phase_now == phase_next and current_green_time < max_green:
            traci.trafficlight.setPhase('J1',phase_now)
            traci.trafficlight.setPhaseDuration('J1',unit_green)
            current_green_time = current_green_time + unit_green
            next_choice_time = current_time + unit_green
        elif (phase_now == phase_next and current_green_time >= max_green) or (phase_now != phase_next):
            #次分支维护相位切换的黄灯时间
            if phase_now + 1 <= 10:
                yellow_phase = phase_now + 1
            else:
                raise Exception(f'phase_now = {phase_now} Error!')
            #切黄灯
            traci.trafficlight.setPhase('J1',yellow_phase)
            traci.trafficlight.setPhaseDuration('J1',3)
            for _ in range(3): traci.simulationStep()
            #切下一个相位
            traci.trafficlight.setPhase('J1',phase_next)
            traci.trafficlight.setPhaseDuration('J1',min_green) 
            current_green_time = min_green - 1
            next_choice_time = current_time + min_green + 3
        return current_green_time,next_choice_time
    
    def actuated_phase_next(self,threshold = 20):
        current_phase = traci.trafficlight.getPhase("J1")
        if current_phase != 0 and current_phase != 3 and current_phase != 6 and current_phase != 9:
            current_phase -= 1
        phase_list = [0, 3, 6, 9]
        lane_list = [('E0_1','E2_1'),('E0_2','E2_2'),('E1_1','E3_1'),('E1_2','E3_2')]
        phase_lane_dict = dict(zip(phase_list,lane_list))
        phase_pressure = traci.lane.getLastStepVehicleNumber(phase_lane_dict[current_phase][0]) + traci.lane.getLastStepVehicleNumber(phase_lane_dict[current_phase][1])
        if phase_pressure >= threshold :
            return current_phase
        else:
            next_phase = current_phase + 3
            return next_phase
    
    def actuated_step(self):
        assert self.launch_env_flag
        current_green_time = 15
        next_choice_time = 200
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            #前200秒用定周期
            if current_time < 200 :
                continue
            else:
                if current_time == next_choice_time:
                    phase_next = self.actuated_phase_next(threshold = 20)
                    phase_now = traci.trafficlight.getPhase("J1")
                    current_time = traci.simulation.getTime()
                    current_green_time,next_choice_time = self.actuated_control(
                        actuated_phase_next = phase_next,
                        min_green = 10,
                        max_green = 45,
                        unit_green= 5,
                        current_green_time = current_green_time,
                        current_time = current_time,
                        phase_now = phase_now)
                else:
                    continue
                
    def debug_step(self):
        assert self.launch_env_flag    
        while traci.simulation.getMinExpectedNumber() > 0:
            phase = traci.trafficlight.getPhase("J1")
            phase_dur = traci.trafficlight.getPhaseDuration("J1")
            next_phase = traci.trafficlight.getNextSwitch("J1")
            control_lanes = traci.trafficlight.getControlledLanes("J1")
            control_links = traci.trafficlight.getControlledLinks("J1")
            print(f"control_lanes = {control_lanes},control_links = {control_links}")
            traci.simulationStep()
        
        
    def runSimulation(self):
        """开始模拟
        """
        self.launchEnv()  
        #self.actuated_step()  
        #self.debug_step()
        self.step()
        self.close()  

if __name__ == '__main__':
    sumo_sim = Simulation(net_config=r'sumo_rl\nets\2x2grid\2x2.net.xml',
                          route_config=r'sumo_rl\nets\2x2grid\2x2.rou.xml',
                          GUI=True)
    sumo_sim.runSimulation()   
    
# "--statistics-output" will in "webster.xml"