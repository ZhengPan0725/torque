import asyncio
import plotly.express as px
from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import numpy as np
import re
from skspatial.objects import Plane

# Define UI components
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file_input", "Upload FHI-aims output files", multiple=True),
        ui.input_text("atom_selection", "Select atoms (e.g., 1, 3-5, 8)", placeholder="Enter atoms"),
        ui.input_switch("input_axis", "Custom Axis Input", True),
        ui.output_ui("axis_input_panel"),
        ui.input_action_button("calculate_button", "Calculate Torque"),
    ),
    ui.card(
        output_widget("geometry_plot"),
        ui.output_text_verbatim("momentum_output"),
    )
)

# Helper function to parse atom selection input
def parse_atom_selection(selection_str):
    atoms = []
    ranges = selection_str.split(',')
    for r in ranges:
        if '-' in r:
            try:
                start, end = map(int, r.split('-'))
                atoms.extend(range(start, end + 1))
            except ValueError:
                return []  # Return empty list if parsing fails
        else:
            try:
                atoms.append(int(r))
            except ValueError:
                return []  # Return empty list if parsing fails
    return atoms

# Helper function to locate and extract the first geometry data from a file
def extract_geometry(file_content):
    geometry_block = []
    parsing_geometry = False

    lines = file_content.splitlines()
    for idx, line in enumerate(lines):
        if "Parsing geometry.in (first pass over file, find array dimensions only)" in line and not geometry_block:
            parsing_geometry = True
            # Skip empty lines before geometry data starts
            while idx + 1 < len(lines) and lines[idx + 1].strip() == "":
                idx += 1
            continue

        if parsing_geometry:
            if line.lstrip().startswith("atom"):
                geometry_block.append(line)
            elif line.strip() == "" and geometry_block:
                parsing_geometry = False
                break  # Stop after finding the first complete geometry block

    return geometry_block

# Helper function to locate and extract the first force data from a file
def extract_forces(file_content):
    forces_block = []
    parsing_forces = False
    line_detected = False

    lines = file_content.splitlines()
    for idx, line in enumerate(lines):
        line_detected = "Total atomic forces (unitary forces cleaned) [eV/Ang]:" in line or "Total atomic forces (unitary forces were cleaned, then relaxation constraints were applied) [eV/Ang]:" in line
        if line_detected and not forces_block:
            parsing_forces = True
            # Skip empty lines before forces data starts
            while idx + 1 < len(lines) and lines[idx + 1].strip() == "":
                idx += 1
            continue

        if parsing_forces:
            if line.lstrip().startswith("|"):
                forces_block.append(line)
            # Stop if we encounter an empty line or if the number of lines matches the number of atoms
            if line.strip() == "" or len(forces_block) == len(extract_geometry(file_content)):
                parsing_forces = False
                break

    return forces_block

# Function to calculate the total momentum
def calculate_momentum(geometry, forces, selected_atoms, custom_axis = False, chosen_id = None):
    # Calculate the position of Cr and the center of selected six Carbon atoms
    if custom_axis:
        chosen_positions = []
        for i, line in enumerate(geometry):
            if i + 1 in chosen_id:
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                chosen_positions.append(np.array([x, y, z]))
        if len(chosen_positions) != 2:
            raise ValueError("chosen_position is not 2!")
        else:
            axis_direction = chosen_positions[0] - chosen_positions[1]
            cr_position = chosen_positions[0]
    else:
        cr_position = None
        carbon_positions = []

        for i, line in enumerate(geometry):
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            element = parts[4]
            if element == "Cr":
                cr_position = np.array([x, y, z])
            elif element == "C" and i + 1 in selected_atoms:
                carbon_positions.append(np.array([x, y, z]))

        if len(carbon_positions) < 6:
            raise ValueError("Not enough Carbon atoms for axis calculation.")

        if cr_position is None:
            raise ValueError("Cr needed to specify the axis")

        # Calculate the center of the six Carbon atoms
        carbon_center = np.mean(carbon_positions, axis=0)
        axis_direction = carbon_center - cr_position

    axis_direction /= np.linalg.norm(axis_direction)  # Normalize the axis direction

    # Calculate the total momentum with respect to the axis
    total_momentum = 0.0
    for i, force_line in enumerate(forces):
        if i + 1 in selected_atoms:
            _, fx, fy, fz = map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", force_line))
            force_vector = np.array([fx, fy, fz])
            position_vector = np.array([float(x) for x in geometry[i].split()[1:4]]) - cr_position
            # Calculate the component of the momentum perpendicular to the axis
            perpendicular_component = np.cross(position_vector, force_vector)
            momentum_along_axis = np.dot(perpendicular_component, axis_direction)
            total_momentum += momentum_along_axis
    return total_momentum

# Server logic
def server(input, output, session):

    @render_widget
    @reactive.event(input.calculate_button)
    def geometry_plot():
        if input.file_input() is None:
            return px.scatter_3d(width = 800, height = 600)

        # Load the first file for simplicity (extendable to multiple files)
        file_info = input.file_input()[0]
        with open(file_info["datapath"], 'r') as f: file_content = f.read()
        geometry = extract_geometry(file_content)

        # Parse atom positions
        if not geometry:
            return px.scatter_3d(width = 800, height = 600)

        atoms = []
        for line in geometry:
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            element = parts[4]
            atoms.append((x, y, z, element))

        # Create 3D scatter plot of atoms
        df = {'x': [atom[0] for atom in atoms], 'y': [atom[1] for atom in atoms], 'z': [atom[2] for atom in atoms], 'element': [atom[3] for atom in atoms]}
        df_t = np.array(list([df['x'], df['y'], df['z']]))
        axis_setting = dict(nticks = 6, range = [df_t.min().min() - 1, df_t.max().max() + 1])

        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            color='element', title='Geometry Plot',
                            width = 800, height = 600)
        fig.update_layout(scene=dict(xaxis = axis_setting,
                                yaxis = axis_setting,
                                zaxis = axis_setting))

        return fig

    @output
    @render.text
    @reactive.event(input.calculate_button)
    async def momentum_output():
        if input.file_input() is None or input.atom_selection() == "":
            return "Please upload files and select atoms."

        custom_axis = input.input_axis()
        # Load the first file for simplicity (extendable to multiple files)
        file_info = input.file_input()[0]
        with open(file_info["datapath"], 'r') as f: file_content = f.read()
        geometry = extract_geometry(file_content)
        forces = extract_forces(file_content)
        selected_atoms = parse_atom_selection(input.atom_selection())

        if not selected_atoms:
            return "Invalid atom selection. Please enter valid atom indices."

        # Calculate total momentum
        if not forces:
            return "No forces data found."

        if custom_axis:
            chosen_id = [int(x) for x in input.chosen_id().split(",")]
        else:
            chosen_id = None
        try:
            total_momentum = calculate_momentum(geometry, forces, selected_atoms, custom_axis, chosen_id)
            return f"Total Momentum: {total_momentum} (eV)"
        except ValueError as e:
            return str(e)

    @render.ui
    @reactive.event(input.input_axis)
    def axis_input_panel():
        if input.input_axis():
            return ui.input_text("chosen_id", "Enter two number(e.g. 1,2)")

# Create the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
