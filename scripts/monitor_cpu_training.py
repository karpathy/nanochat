#!/usr/bin/env python3
"""
CPU Training Monitor for i5 processors
Monitors system resources during nanochat training
"""

import time
import psutil
import threading
import sys
import os

def monitor_system(stop_event, log_file="cpu_monitor.log"):
    """Monitor CPU, memory, and temperature during training"""
    print(f"Starting CPU monitoring... Logging to {log_file}")
    
    with open(log_file, 'w') as f:
        f.write("timestamp,cpu_percent,memory_gb,memory_percent,cpu_temp\n")
        
        while not stop_event.is_set():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_gb = memory.used / (1024**3)
                memory_percent = memory.percent
                
                # CPU temperature (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    cpu_temp = "N/A"
                    if 'coretemp' in temps:
                        cpu_temp = max(temp.current for temp in temps['coretemp'])
                    elif 'cpu_thermal' in temps:
                        cpu_temp = temps['cpu_thermal'][0].current
                except:
                    cpu_temp = "N/A"
                
                # Log data
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"{timestamp},{cpu_percent:.1f},{memory_gb:.1f},{memory_percent:.1f},{cpu_temp}"
                f.write(log_line + "\n")
                f.flush()
                
                # Console output every 30 seconds
                if int(time.time()) % 30 == 0:
                    print(f"[{timestamp}] CPU: {cpu_percent:.1f}%, Memory: {memory_gb:.1f}GB ({memory_percent:.1f}%), Temp: {cpu_temp}°C")
                
                # Warning if resources are high
                if cpu_percent > 90:
                    print(f"⚠️  HIGH CPU USAGE: {cpu_percent:.1f}%")
                if memory_percent > 85:
                    print(f"⚠️  HIGH MEMORY USAGE: {memory_percent:.1f}%")
                if cpu_temp != "N/A" and cpu_temp > 80:
                    print(f"⚠️  HIGH CPU TEMP: {cpu_temp}°C")
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)

def estimate_training_time(num_iterations, current_iteration=0):
    """Estimate remaining training time based on i5 performance"""
    # Rough estimates for i5 CPU (varies by model and generation)
    iterations_per_hour = 60  # Conservative estimate for small model
    
    remaining = num_iterations - current_iteration
    hours = remaining / iterations_per_hour
    
    if hours < 1:
        return f"{int(hours * 60)} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"

def main():
    print("NanoChat CPU Training Monitor")
    print("=" * 40)
    
    # Check system specs
    print(f"CPU: {psutil.cpu_count()} cores, {psutil.cpu_freq().max:.0f}MHz max")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB total")
    print(f"Python: {sys.version}")
    print()
    
    # Start monitoring in background
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_system, args=(stop_event,))
    monitor_thread.start()
    
    try:
        print("Monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        stop_event.set()
        monitor_thread.join()
        print("Monitor stopped.")

if __name__ == "__main__":
    main()