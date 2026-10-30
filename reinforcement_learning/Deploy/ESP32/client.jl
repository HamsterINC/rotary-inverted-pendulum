import MeshCat
import MeshCatMechanisms
import RigidBodyDynamics as RBD
import RotaryInvertedPendulum as RIP

using Libdl
using LibSerialPort
using LinearAlgebra
using Statistics

const path_to_tinympc_julia = "/Users/henrique/git/Rotary Inverted Pendulum/RotaryInvertedPendulum-julia/tinympc-julia"

# Load the TinyMPC Julia interface
include("$(path_to_tinympc_julia)/tinympc/TinyMPC.jl")
using .TinyMPC

const lib_tinympc_codegen = "$(path_to_tinympc_julia)/generated_code/build/tinympc/libtinympcShared.dylib"

# Define the serial port and baud rate
serial_port = "/dev/cu.usbserial-110"  # Replace with your serial port
baud_rate = 2000000

# Command bytes
const CMD_READY = UInt8(0x01)
const CMD_GET_STATE = UInt8(0x02)
const CMD_SET_TARGET = UInt8(0x03)
const CMD_ENGAGE_MOTOR = UInt8(0x04)
const CMD_DISENGAGE_MOTOR = UInt8(0x05)

# Function to check if the Arduino is ready
function check_arduino_ready(port; retries=3, timeout_seconds=1)
    for attempt in 1:retries
        println("Checking if Arduino is ready... (Attempt $attempt/$retries)")
        LibSerialPort.write(port, [CMD_READY])
        LibSerialPort.flush(port, LibSerialPort.SP_BUF_BOTH)

        response = UInt8[]
        t_timeout = time_ns() + timeout_seconds * 1e9

        while time_ns() < t_timeout
            if LibSerialPort.bytesavailable(port) > 0
                push!(response, LibSerialPort.read(port, UInt8))
                break
            end
        end

        if length(response) == 1 && response[1] == CMD_READY
            println("Arduino is ready!")
            return true
        else
            println("No valid response. Retrying...")
        end

        sleep(0.5)
    end
    println("Failed to get a valid response after $retries attempts.")
    return false
end

# Function to retrieve state from Arduino
# https://github.com/JuliaIO/LibSerialPort.jl/blob/master/test/test-low-level-api.jl
function get_pendulum_state(port)
    LibSerialPort.write(port, [CMD_GET_STATE])  # Send command
    # LibSerialPort.sp_blocking_write(port.ref, [CMD_GET_STATE], 500)  # Send command
    state_data = zeros(UInt8, 12)               # Expect 12 bytes of data (4 + 4 + 4)
    nbytes = LibSerialPort.sp_blocking_read(port.ref, Ref(state_data, 1), 12, 500)  # Read 12 bytes with timeout

    if nbytes == 12
        current_time = reinterpret(Int32, state_data[1:4])[1]  # Microseconds
        motor_position = reinterpret(Float32, state_data[5:8])[1]  # Motor position in radians
        pendulum_position = reinterpret(Float32, state_data[9:12])[1]  # Pendulum position in radians
        return current_time, motor_position, pendulum_position
    else
        println("Error: Expected 12 bytes, but received $nbytes bytes.")
        return nothing
    end
end

# Function to set the motor target position (in radians)
function set_motor_target(port, sm_target_position::Float32)
    # Convert sm_target_position to an array of bytes
    target_bytes = reinterpret(UInt8, [sm_target_position])
    @assert length(target_bytes) == 4
    # Send the command and the target position bytes
    LibSerialPort.write(port, [CMD_SET_TARGET; target_bytes])
    # LibSerialPort.sp_blocking_write(port.ref, [CMD_SET_TARGET; target_bytes], 500)
end

# Function to engage the motor
function engage_motor(port)
    LibSerialPort.write(port, [CMD_ENGAGE_MOTOR])
end

# Function to disengage the motor
function disengage_motor(port)
    LibSerialPort.write(port, [CMD_DISENGAGE_MOTOR])
end

# Main function to interact with the Arduino
function main(; visualise=false)
    # Load the URDF model of the rotary inverted pendulum
    package_path = joinpath(pkgdir(RIP), "..")
    filename = joinpath(package_path, "urdf/model.urdf")
    mechanism = RBD.parse_urdf(filename)

    rip_dynamics, A, B = RIP.linearise_dynamics(mechanism, 1 / 150)

    println()
    println("A, state-transition matrix:")
    display(A)

    println()
    println("B, control matrix:")
    display(B)

    # `Q` is the stage cost for the state. It defines how much to penalise
    # the state for deviating from the reference state at each time step.
    # `R` is the stage cost for the input. It defines how much to penalise
    # the input for deviating from the reference control at each time step.
    Q = Diagonal([10, 1, 10, 1])
    R = Diagonal([1])

    # Riccati recursion on the linearized dynamics
    P = 1 * Q
    K = zeros(1, 4)
    for _ in 1:1000
        P = Q + A' * P * A - A' * P * B * inv(R + B' * P * B) * B' * P * A
        K = inv(R + B' * P * B) * B' * P * A
    end

    println()
    println("K matrix:")
    display(K)

    # The LQR controller
    function lqr_controller(x::Vector)
        return -K * x
    end

    n = 4  # number of state variables
    m = 1  # number of control variables
    num_knots_mpc_horizon = 20

    # Limits on state and control variables
    pos_lim = π/2  # in radians
    vel_lim = 5.0   # in radians per second
    tau_lim = 0.08  # in Newton meters

    # Placeholder for the list of control inputs in the MPC horizon
    u = zeros(Float32, (m, num_knots_mpc_horizon - 1))

    # Use delta because MPC uses the linearized dynamics around upright position
    # Set the reference state to 0 as well as reset
    delta_xref = zeros(Float32, (n, num_knots_mpc_horizon))  # reference state
    @ccall lib_tinympc_codegen.set_xref(delta_xref::Ptr{Float32}, 0::Cint)::Cvoid

    visualise && @info "Visualising flag has been set to true. Opening visualiser..."

    if visualise
        vis = MeshCat.Visualizer()
        open(vis)

        MeshCat.setprop!(vis["/Cameras/default/rotated/<object>"], "fov", 40)

        urdfvisuals = MeshCatMechanisms.URDFVisuals(filename, package_path=[package_path])
        mvis = MeshCatMechanisms.MechanismVisualizer(mechanism, urdfvisuals, vis["model"])

        last_vis_update_time = time_ns()
    end

    port = LibSerialPort.open(serial_port, baud_rate; parity=LibSerialPort.SP_PARITY_NONE)
    LibSerialPort.set_flow_control(port)

    sleep(1)  # Allow time for Arduino to initialize

    if !check_arduino_ready(port)
        println("Error: Arduino not ready. Exiting.")
        LibSerialPort.close(port)
        return
    end

    println("Engaging motor...")
    engage_motor(port)

    println("Starting data retrieval and control loop...")

    start_time = time_ns()
    loop_duration_s = 120  # in seconds
    loop_duration_ns = loop_duration_s * 1e9
    iteration_times = Float64[]
    last_frequency_time = time_ns()

    rate_stats_print = 0.5 * 1e9  # every 0.5 seconds in nanoseconds
    first_loop = true
    initial_arduino_time = 0
    prev_arduino_time = 0
    q_prev = [0, 0]
    sm_target_position = 0

    x_goal = Float64[0, π, 0, 0]

    vis_fps = 60  # frames per second of the visualisation
    vis_update_interval = 1 / vis_fps  # in seconds
    vis_update_interval_ns = vis_update_interval * 1e9

    while time_ns() - start_time < loop_duration_ns
        iteration_start_time = time_ns()

        state = get_pendulum_state(port)
        if state !== nothing
            current_arduino_time, motor_position, pendulum_position = state

            if first_loop
                initial_arduino_time = current_arduino_time
                prev_arduino_time = current_arduino_time
                first_loop = false
            end

            # Convert elapsed time (since last iteration) from microseconds to seconds
            dt = (current_arduino_time - prev_arduino_time) / 1e6

            q = [motor_position, pendulum_position]
            v = q - q_prev
            x = [q; v]

            # Update visualization only if sufficient time has passed
            if visualise && (iteration_start_time - last_vis_update_time) >= vis_update_interval_ns
                RBD.set_configuration!(mvis, q)
                last_vis_update_time = iteration_start_time
            end

            # # Compute target position
            # 𝐴 = 45  # Amplitude of the sine wave (degrees)
            # 𝑓 = 1.0  # Frequency of the sine wave (Hz)
            # 𝑡 = (current_arduino_time - initial_arduino_time) / 1e6  # Convert elapsed time from microseconds to seconds
            # sm_target_position = deg2rad(𝐴) * sin(2π * 𝑓 * 𝑡)  # Sine wave

            # # Compute target position
            # u = lqr_controller(x - x_goal)
            # clamp!(u, -tau_lim, tau_lim)
            # sm_target_position -= u[1]

            # Set initial state from measurement
            delta_x_noise = Vector{Float32}(x - x_goal)
            @ccall lib_tinympc_codegen.set_x0(delta_x_noise::Ptr{Float32}, 0::Cint)::Cvoid  # set the current state
            @ccall lib_tinympc_codegen.call_tiny_solve(0::Cint)::Cvoid                      # solve the problem
            @ccall lib_tinympc_codegen.get_u(u::Ptr{Matrix{Float32}}, 0::Cint)::Cvoid       # get the control input
            clamp!(u, -tau_lim, tau_lim)

            # simulate torque control so we can use the velocity for the stepper motor target
            vars = convert(Vector{Float64}, [x; -u[1]])
            result = rip_dynamics(vars)
            new_sm_vel = result[1] + dt * result[3]
            new_sm_pos = sm_target_position + dt * new_sm_vel
            sm_target_position = new_sm_pos

            # println("u: $u, v: $(result[1]), p: $sm_target_position")

            # Send target position to motor
            sm_target_position = clamp(sm_target_position, -pos_lim, pos_lim)  # clamp target pos
            set_motor_target(port, Float32(sm_target_position))

            prev_arduino_time = current_arduino_time
            q_prev = q

            # println("Time: $(current_arduino_time) µs, Motor: $(motor_position) rad, Pendulum: $(pendulum_position) rad, Target: $(sm_target_position) rad")
        end

        iteration_duration = (time_ns() - iteration_start_time) / 1e9  # Duration in seconds
        push!(iteration_times, iteration_duration)

        if time_ns() - last_frequency_time >= rate_stats_print
            if !isempty(iteration_times)
                avg_freq = 1 / mean(iteration_times)
                println("Average loop frequency: $(round(avg_freq, digits=2)) Hz")
                empty!(iteration_times)  # Reset for next calculation
            end
            last_frequency_time = time_ns()
        end
    end

    println("Disengaging motor...")
    disengage_motor(port)

    println("Data retrieval and control complete.")
    LibSerialPort.close(port)

    if visualise
        MeshCat.close(vis)
    end
end

# Execute the main function if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    visualise = "--visualise" in ARGS
    main(visualise=visualise)
end
