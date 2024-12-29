import os
import sys
import traci
import numpy as np
from typing import Tuple, Dict, Any

class RampMeterEnv:
    """SUMO Environment for ramp metering control"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sumo_binary = config.get('sumo_binary', 'sumo-gui' if config.get('gui', False) else 'sumo')
        self.net_file = config.get('net_file')
        self.route_file = config.get('route_file')
        
        # State and action space
        self.observation_space_size = 6  # [mainline_density, ramp_queue, avg_speed, etc.]
        self.action_space_size = 2       # [red_light, green_light]
        
        # Traffic light ID
        self.tl_id = "merge"  # Changed from "ramp_tl" to match our network file
        self.simulation_step = 0
        
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state"""
        self.close()  # Close any existing connection
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise EnvironmentError("Please declare environment variable 'SUMO_HOME'")
            
        sumo_cmd = [self.sumo_binary,
                    '-n', self.net_file,
                    '-r', self.route_file,
                    '--no-warnings',
                    '--start']
                    
        try:
            traci.start(sumo_cmd)
            self.simulation_step = 0
            return self._get_state()
        except traci.exceptions.TraCIException as e:
            print(f"Error starting SUMO: {e}")
            self.close()
            raise
            
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return new state, reward, done, info"""
        # Apply action (0: Red, 1: Green)
        self._apply_action(action)
        
        # Simulate one step
        traci.simulationStep()
        self.simulation_step += 1
        
        # Get new state and calculate reward
        state = self._get_state()
        reward = self._calculate_reward()
        
        # Check if simulation is done
        done = traci.simulation.getMinExpectedNumber() <= 0 or self.simulation_step >= 3600
        
        # Collect additional info
        info = {
            'waiting_time': self._get_average_waiting_time(),
            'vehicles_passed': self._get_vehicles_passed(),
            'avg_speed': self._get_average_speed()
        }
        
        return state, reward, done, info
        
    def _get_state(self) -> np.ndarray:
        """Get current state of the environment"""
        try:
            # Get vehicle counts
            mainline_vehicles_in = len(traci.edge.getLastStepVehicleIDs("highway_in"))
            mainline_vehicles_out = len(traci.edge.getLastStepVehicleIDs("highway_out"))
            ramp_vehicles = len(traci.edge.getLastStepVehicleIDs("ramp"))
            
            # Get speeds
            mainline_speed_in = traci.edge.getLastStepMeanSpeed("highway_in") * 3.6  # Convert to km/h
            mainline_speed_out = traci.edge.getLastStepMeanSpeed("highway_out") * 3.6
            ramp_speed = traci.edge.getLastStepMeanSpeed("ramp") * 3.6
            
            # Get queue length on ramp
            ramp_queue = len([v for v in traci.edge.getLastStepVehicleIDs("ramp") 
                            if traci.vehicle.getSpeed(v) < 0.1])
            
            # Calculate density (vehicles per km)
            # Use fixed lengths since getLength is not available
            highway_in_length = 500  # meters from our network file
            highway_out_length = 500
            total_length_km = (highway_in_length + highway_out_length) / 1000
            
            mainline_density = (mainline_vehicles_in + mainline_vehicles_out) / total_length_km
            
            return np.array([
                mainline_density,
                ramp_queue,
                (mainline_speed_in + mainline_speed_out) / 2,
                ramp_speed,
                mainline_vehicles_in + mainline_vehicles_out,
                ramp_vehicles
            ])
        except traci.exceptions.TraCIException as e:
            print(f"TraCI Exception: {e}")
            return np.zeros(self.observation_space_size)
            
    def _apply_action(self, action: int):
        """Apply the selected action to the traffic light"""
        try:
            if action == 0:  # Red light
                traci.trafficlight.setPhase(self.tl_id, 2)  # Red phase
            else:  # Green light
                traci.trafficlight.setPhase(self.tl_id, 0)  # Green phase
        except traci.exceptions.TraCIException as e:
            print(f"Error applying action: {e}")
            
    def _calculate_reward(self) -> float:
        """Calculate reward based on traffic metrics"""
        try:
            # Get average speeds
            mainline_speed = (traci.edge.getLastStepMeanSpeed("highway_in") +
                            traci.edge.getLastStepMeanSpeed("highway_out")) / 2 * 3.6
            
            # Get queue lengths
            ramp_queue = len([v for v in traci.edge.getLastStepVehicleIDs("ramp")
                            if traci.vehicle.getSpeed(v) < 0.1])
            
            # Calculate reward components
            speed_reward = mainline_speed / 50.0  # Normalize by desired speed
            queue_penalty = -0.1 * ramp_queue
            
            return speed_reward + queue_penalty
            
        except traci.exceptions.TraCIException as e:
            print(f"Error calculating reward: {e}")
            return 0.0
            
    def _get_average_waiting_time(self) -> float:
        """Calculate average waiting time of vehicles"""
        try:
            total_waiting_time = 0
            vehicle_count = 0
            
            for edge in ["highway_in", "highway_out", "ramp"]:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                for vehicle in vehicles:
                    total_waiting_time += traci.vehicle.getWaitingTime(vehicle)
                    vehicle_count += 1
                    
            return total_waiting_time / max(1, vehicle_count)
            
        except traci.exceptions.TraCIException as e:
            print(f"Error calculating waiting time: {e}")
            return 0.0
            
    def _get_vehicles_passed(self) -> int:
        """Count vehicles that have passed through the merge area"""
        try:
            return len(traci.edge.getLastStepVehicleIDs("highway_out"))
        except traci.exceptions.TraCIException as e:
            print(f"Error counting vehicles: {e}")
            return 0
            
    def _get_average_speed(self) -> float:
        """Calculate average speed of all vehicles"""
        try:
            total_speed = 0
            vehicle_count = 0
            
            for edge in ["highway_in", "highway_out", "ramp"]:
                total_speed += traci.edge.getLastStepMeanSpeed(edge) * 3.6  # Convert to km/h
                if len(traci.edge.getLastStepVehicleIDs(edge)) > 0:
                    vehicle_count += 1
                    
            return total_speed / max(1, vehicle_count)
            
        except traci.exceptions.TraCIException as e:
            print(f"Error calculating average speed: {e}")
            return 0.0
            
    def close(self):
        """Close the SUMO simulation"""
        try:
            traci.close()
            if hasattr(traci, '_connections'):
                for conn in traci._connections.values():
                    conn.close()
            traci._connections = {}
        except Exception as e:
            print(f"Error closing TRACI: {e}")
