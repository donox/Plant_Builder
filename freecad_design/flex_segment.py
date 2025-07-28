import numpy as np
import csv
from .wafer import Wafer
import FreeCAD
import FreeCADGui
from .utilities import position_to_str


class FlexSegment(object):
    def __init__(self, prefix,  show_lcs, temp_file, to_build, rotate_segment, trace=None):
        self.doc = FreeCAD.ActiveDocument
        self.gui = FreeCADGui
        self.prefix = prefix + "_"
        self.rotate_segment = rotate_segment
        self.trace = trace
        self.wafer_count = 0
        self.wafer_list = []            # This is empty for segments that were reconstructed from the model tree
        self.show_lcs = show_lcs
        self.temp_file = temp_file
        self.segment_type = None
        self.to_build = to_build    # if False, use existing structure
        self.segment_object = None      # THIS WAS HELIX - DOES NOT EXIST HERE?
        self.transform_to_top = None  # transform from base of segment t0 top of segment
        if self.to_build:
            self.remove_prior_version()
        self.lcs_base = self.doc.addObject('PartDesign::CoordinateSystem', self.prefix + "_lcs_base")
        self.lcs_top = self.doc.addObject('PartDesign::CoordinateSystem', self.prefix + "_lcs_top")
        self.lcs_group = self.doc.addObject("App::DocumentObjectGroup", self.prefix + "_lcs_group")
        # Create main segment group for organization
        self.main_group = self.doc.addObject("App::DocumentObjectGroup", self.prefix + "segment_group")
        self.visualization_group = self.doc.addObject("App::DocumentObjectGroup", self.prefix + "visualization")
        self.main_group.addObject(self.visualization_group)
        self.main_group.addObject(self.lcs_group)
        self.transform_callbacks = []
        self._setup_transform_properties()
        self.already_relocated = False  # Track relocation status


    def add_wafer(self, lift, rotation, cylinder_diameter, outside_height, wafer_type="EE"):
        # Make wafer at base and move after creation.  Creating at the target location seems to confuse OCC
        # causing some wafer to be constructed by lofting to the wrong side of the target ellipse.
        self.wafer_count += 1
        name_base = self.prefix + str(self.wafer_count)
        wafer_name = name_base + "_w"
        wafer = Wafer(FreeCAD, self.gui, self.prefix, wafer_type=wafer_type)
        wafer.set_parameters(lift, rotation, cylinder_diameter, outside_height, wafer_type="EE")
        lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', name_base + "_1lcs")
        lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', name_base + "_2lcs")
        self.lcs_group.addObjects([lcs1, lcs2])
        wafer.lift_lcs(lcs2, wafer_type)
        matrix = lcs2.Placement.toMatrix()
        matrix.rotateZ(-rotation)
        lcs2.Placement = FreeCAD.Placement(matrix)
        if not self.show_lcs:
            lcs1.Visibility = False
            lcs2.Visibility = False
        wafer.make_wafer_from_lcs(lcs1, lcs2, cylinder_diameter, wafer_name)
        # print(f"Wafer {wafer_name} angle (top ellipse) to X-Y plane: {np.round(wafer.get_angle(), 3)}")

        lcs1.Placement = self.lcs_top.Placement
        lcs2.Placement = self.lcs_top.Placement.multiply(lcs2.Placement)
        wafer_loft = wafer.get_wafer()
        wafer_loft.Placement = lcs1.Placement
        self.wafer_list.append(wafer)
        self.lcs_top.Placement = lcs2.Placement

    def add_wafer_rectangle(self, lift, rotation, long_side, outside_height, wafer_type="EE"):
        # Make wafer at base and move after creation.  Creating at the target location seems to confuse OCC
        # causing some wafer to be constructed by lofting to the wrong side of the target ellipse.
        self.wafer_count += 1
        name_base = self.prefix + str(self.wafer_count)
        wafer_name = name_base + "_w"
        wafer = Wafer(FreeCAD, self.gui, self.prefix, wafer_type=wafer_type)
        wafer.set_parameters(lift, rotation, long_side, outside_height, wafer_type="EE")
        lcs1 = self.doc.addObject('PartDesign::CoordinateSystem', name_base + "_1lcs")
        lcs2 = self.doc.addObject('PartDesign::CoordinateSystem', name_base + "_2lcs")
        self.lcs_group.addObjects([lcs1, lcs2])
        wafer.lift_lcs(lcs2, wafer_type)
        # matrix = lcs2.Placement.toMatrix()
        # matrix.rotateZ(-rotation)
        # lcs2.Placement = FreeCAD.Placement(matrix)
        if not self.show_lcs:
            lcs1.Visibility = False
            lcs2.Visibility = False
        wafer.make_rectangle_wafer_from_lcs(lcs1, lcs2, long_side, long_side, wafer_name)
        # print(f"Wafer {wafer_name} angle (top ellipse) to X-Y plane: {np.round(wafer.get_angle(), 3)}")
        matrix = lcs2.Placement.toMatrix()
        matrix.rotateZ(-rotation)
        lcs2.Placement = FreeCAD.Placement(matrix)

        lcs1.Placement = self.lcs_top.Placement
        lcs2.Placement = self.lcs_top.Placement.multiply(lcs2.Placement)
        wafer_loft = wafer.get_wafer()
        wafer_loft.Placement = lcs1.Placement
        self.wafer_list.append(wafer)
        self.lcs_top.Placement = lcs2.Placement

    def get_segment_name(self):
        return self.prefix[:-1]    # strip trailing underscore

    def get_segment_object(self):
        return self.segment_object

    def get_wafer_count(self):
        return self.wafer_count

    def get_wafer_parameters(self):
        raise NotImplementedError(f"need to identify specific wafer and get from there")
        if self.lift_angle != 0:  # leave as simple cylinder if zero
            la = self.lift_angle / 2
            oh = self.outside_height / 2
            # Assume origin at center of ellipse, x-axis along major axis, positive to outside.
            self.helix_radius = oh / np.math.tan(la)
            # print(f"SET RADIUS: {self.helix_radius}, Lift: {self.lift_angle}, Height: {self.outside_height}")
            self.inside_height = (self.helix_radius - self.cylinder_diameter) * np.math.tan(la) * 2

    def get_lcs_top(self):
        # print(f"LCS TOP: {self.lcs_top.Label}")
        return self.lcs_top

    def get_lcs_base(self):
        if not self.lcs_base:
            raise ValueError(f"lcs_base not set")
        return self.lcs_base

    def get_transform_to_top(self):    # Does this need changing
        if self.to_build:
            if not self.transform_to_top:
                # print(f"TO TOP: Base: {self.lcs_base.Placement}. Top: {self.lcs_top.Placement}")
                self.transform_to_top = self.make_transform_align(self.lcs_base, self.lcs_top)
            return self.transform_to_top
        else:
            print(f"NO TRANSFORM: {self.get_segment_name()}, BUILD? {self.to_build}")
            raise ValueError("Segment has no valid transform to top as it was created in prior run.")

    def fuse_wafers(self):
        name = self.get_segment_name()
        if len(self.wafer_list) > 1:
            fuse = self.doc.addObject("Part::MultiFuse", name + "FusedResult")
            fuse.Shapes = [x.wafer for x in self.wafer_list]
        elif len(self.wafer_list) == 1:
            fuse = self.wafer_list[0].wafer
            fuse.Label = name + "FusedResult"
        else:
            raise ValueError("Zero Length Wafer List when building helix")
        fuse.Visibility = True
        fuse.ViewObject.DisplayMode = "Shaded"
        fuse.Placement = self.lcs_base.Placement
        self.segment_object = fuse
        # Add the fused result to the main group
        if self.segment_object:
            self.main_group.addObject(self.segment_object)
        self.doc.recompute()

    def get_segment_rotation(self):
        return self.rotate_segment

    def move_content(self, transform):
        if self.segment_object:
            pl = self.segment_object.Placement
            self.segment_object.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
            pl = self.lcs_top.Placement
            self.lcs_top.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
            pl = self.lcs_base.Placement
            self.lcs_base.Placement = pl.multiply(pl.inverse()).multiply(transform).multiply(pl)
            # print(f"LABELS: {self.segment_object.Label}, {self.lcs_base.Label}, {self.lcs_top.Label}")
            # self.trace("MOVE", self.prefix, self.lcs_top.Label, self.lcs_top.Placement)
        else:
            print(f"NO CONTENT: {self.get_segment_name()} has no content to move.")

    def move_content_to_zero(self, transform):
        """Relocate to a zero base corresponding to a new build.  Transform is lcs_base.inverse()"""
        self.move_content(transform)

    def remove_prior_version(self):
        # TODO: not do so if making cut list???
        name = self.prefix + ".+"
        doc = FreeCAD.activeDocument()
        doc_list = doc.findObjects(Name=name)  # remove prior occurrence of set being built
        for item in doc_list:
            if item.Label != 'Parms_Master':
                doc.removeObject(item.Name)

    def make_cut_list(self, segment_no, cuts_file):
        parm_str = f"\n\nCut list for segment: {segment_no}\n"
        # parm_str += f"Lift Angle: {np.round(np.rad2deg(self.lift_angle), 2)} degrees\n"
        # parm_str += f"Rotation Angle: {np.rad2deg(self.rotation_angle)} degrees\n"
        # parm_str += f"Outside Wafer Height: {np.round(self.outside_height, 2)} in\n"
        # if self.inside_height:
        #     parm_str += f"Inside Wafer Height: {np.round(self.inside_height, 2)} in\n"
        # else:
        #     parm_str += f"Inside Wafer Height: NONE\n"
        # parm_str += f"Cylinder Diameter: {np.round(self.cylinder_diameter, 2)} in\n"
        # if self.helix_radius:
        #     parm_str += f"Helix Radius: \t{np.round(self.helix_radius, 2)} in\n"
        # else:
        #     parm_str += f"Helix Radius: NONE\n"
        # cuts_file.write(parm_str)
        # cuts_file.write(f"Wafer Count: {self.wafer_count}\n\n")
        # try:
        #     step_size = np.rad2deg(self.rotation_angle)
        # except Exception as e:
        #     nbr_rotations = None
        #     step_size = 0

        current_position = 0  # Angular position of index (0-360) when rotating cylinder for cutting
        s1 = "Step: "
        s2 = " at position: "
        for i, wafer in enumerate(self.wafer_list):
            ra = wafer.get_rotation_angle()
            if ra is not None:
                ra = np.round(np.rad2deg(ra), 2)
            else:
                ra = 0
            la = wafer.get_lift_angle()
            if la:
                la = np.round(np.rad2deg(la), 2)
            else:
                la = "NA"
            oh = wafer.get_outside_height()
            if oh:
                oh = np.round(oh, 2)
            else:
                oh = "NA"
            str1 = str(i)
            if len(str1) == 1:
                str1 = s1 + ' ' + str1      # account for number of steps (max <= 99)
            else:
                str1 = s1 + str1
            str2 = str(current_position)
            if len(str2) == 1:
                str2 = s2 + ' ' + str2
            else:
                str2 = s2 + str2
            str3 = f"\tLift: {la}\tRotate: {ra}\tHeight: {oh}"
            cuts_file.write(f"{str1} {str2} {str3};    Done: _____\n")
            current_position = int((current_position - ra + 180) % 360)

    def print_construction_list(self, segment_no, cons_file, global_placement, find_min_max):
        """Print list giving local and global position and orientation of each wafer."""
        parm_str = f"\nConstruction list for segment: {self.get_segment_name()}\n"
        # if self.inside_height:
        #     parm_str += f"Inside Wafer Height: {position_to_str(self.inside_height)} in\n"
        # else:
        #     parm_str += f"Inside Wafer Height: NONE\n"
        # parm_str += f"Cylinder Diameter:{position_to_str(self.cylinder_diameter)} in\n"
        parm_str += f"Segment Rotation: \t{position_to_str(self.rotate_segment)} in\n"
        parm_str += f"\n\tNote 'angle' below is angle of top wafer surface to X-Y plane.\n"
        cons_file.write(parm_str)
        cons_file.write(f"Wafer Count: {self.wafer_count}\n\n")

        if not self.wafer_list:
            cons_file.write(f"This segment was reconstructed thus there is no wafer list")
            return None
        for wafer_num, wafer in enumerate(self.wafer_list):
            # if wafer_num:
            #     prior_top = top_lcs_place
            top_lcs_place = wafer.get_top().Placement
            global_loc = global_placement.multiply(top_lcs_place)
            num_str = str(wafer_num + 1)       # Make one-based for conformance with  in shop
            local_x = position_to_str(top_lcs_place.Base.x)
            global_x = position_to_str(global_loc.Base.x)
            local_y = position_to_str(top_lcs_place.Base.y)
            global_y = position_to_str(global_loc.Base.y)
            local_z = position_to_str(top_lcs_place.Base.z)
            global_z = position_to_str(global_loc.Base.z)
            find_min_max(global_loc.Base)
            # Seems to be a long way around to get the lcs z-axis's vector
            lcso = FreeCAD.activeDocument().getObject(wafer.get_lcs_top().Label)
            vec = FreeCAD.Vector(0, 0, 1)
            vec2 = lcso.getGlobalPlacement().Rotation.multVec(vec)
            angle = vec2.getAngle(vec)
            str1 = f"Wafer: {num_str}\tat Local:  [{local_x:{5}}, {local_y}, {local_z}],"
            str1 += f" with Angle: {np.round(np.rad2deg(angle), 1)}\n"
            str1 += f"\t\tat Global: [{global_x}, {global_y}, {global_z}]\n"
            cons_file.write(str1)
        return global_loc

    @staticmethod
    def make_transform_align(object_1, object_2):
        """Create transform that will move an object by the same relative positions of two input objects"""
        l1 = object_1.Placement
        l2 = object_2.Placement
        tr = l1.inverse().multiply(l2)
        # print(f"MOVE_S: {object_1.Label}, {object_2.Label}, {tr}")
        return tr

    def _setup_transform_properties(self):
        """Add custom properties to store transformation data."""
        if not hasattr(self.lcs_base, 'AppliedTransformMatrix'):
            # Store the 4x4 transformation matrix as a string (FreeCAD can't store Matrix directly)
            self.lcs_base.addProperty("App::PropertyString", "AppliedTransformMatrix",
                                      "Transform", "Applied transformation matrix as string")
            self.lcs_base.addProperty("App::PropertyStringList", "CurveVerticesGroups",
                                      "Transform", "Names of associated curve vertex groups")
            self.lcs_base.addProperty("App::PropertyBool", "HasStoredTransform",
                                      "Transform", "Whether this segment has a stored transform")

        # Initialize
        self.lcs_base.HasStoredTransform = False
        self.lcs_base.CurveVerticesGroups = []

    def store_applied_transform(self, transform):
        """Store transform in FreeCAD object properties for persistence."""
        if transform:
            # Convert FreeCAD.Placement to matrix and store as string
            matrix = transform.toMatrix()
            matrix_str = ",".join([str(matrix.A[i]) for i in range(16)])

            self.lcs_base.AppliedTransformMatrix = matrix_str
            self.lcs_base.HasStoredTransform = True

            print(f"Stored transform matrix in FreeCAD object: {self.lcs_base.Label}")
        else:
            self.lcs_base.HasStoredTransform = False

    def get_stored_transform(self):
        """Retrieve stored transform from FreeCAD object properties."""
        if self.lcs_base.HasStoredTransform and self.lcs_base.AppliedTransformMatrix:
            try:
                # Parse matrix string back to FreeCAD.Matrix
                matrix_values = [float(x) for x in self.lcs_base.AppliedTransformMatrix.split(',')]
                matrix = FreeCAD.Matrix()
                for i in range(16):
                    matrix.A[i] = matrix_values[i]

                # Convert matrix back to Placement
                return FreeCAD.Placement(matrix)
            except Exception as e:
                print(f"Error retrieving stored transform: {e}")
                return None
        return None

    def register_curve_vertices_group(self, group_name):
        """Register a curve vertices group with this segment."""
        # Store the group name
        current_groups = list(self.lcs_base.CurveVerticesGroups)
        if group_name not in current_groups:
            current_groups.append(group_name)
            self.lcs_base.CurveVerticesGroups = current_groups

        # Force document recompute before trying to find objects
        self.doc.recompute()

        # Find and move the vertex group into this segment's visualization group
        vertex_groups = self.doc.getObjectsByLabel(group_name)
        if vertex_groups:
            vertex_group = vertex_groups[0]
            try:
                # Remove from document root if it's there
                if hasattr(vertex_group, 'removeFromDocument'):
                    pass  # It's already in the document

                # Add to visualization group
                self.visualization_group.addObject(vertex_group)
                print(f"Successfully moved vertex group '{group_name}' to visualization group")
            except Exception as e:
                print(f"Failed to move vertex group to visualization group: {e}")
        else:
            print(f"Warning: Could not find vertex group '{group_name}' to move to visualization group")

    def register_arrow(self, arrow_obj):
        """Register an arrow with this segment's visualization group."""
        if arrow_obj:
            self.visualization_group.addObject(arrow_obj)
            print(f"Added arrow '{arrow_obj.Label}' to segment visualization group")

    def register_transform_callback(self, callback_func):
        """Register a function to be called when segment is transformed."""
        self.transform_callbacks.append(callback_func)

    def move_to_top(self, transform):
        """Apply transform to reposition segment."""
        self.move_content(transform)

        # Store transform in FreeCAD properties for persistence
        self.store_applied_transform(transform)

        # Apply to registered curve vertices
        self._transform_registered_vertices(transform)

        self.doc.recompute()
        FreeCADGui.updateGui()
        FreeCADGui.SendMsgToActiveView("ViewFit")

        # Call Python callbacks
        for callback in self.transform_callbacks:
            callback(transform)

    def _transform_registered_vertices(self, transform):
        """Transform all registered curve vertex groups."""
        for group_name in self.lcs_base.CurveVerticesGroups:
            curve_groups = self.doc.getObjectsByLabel(group_name)
            if curve_groups:
                vertex_group_obj = curve_groups[0]

                # Transform each individual vertex in the group
                for vertex_obj in vertex_group_obj.OutList:
                    if hasattr(vertex_obj, 'Placement'):
                        current_placement = vertex_obj.Placement
                        new_placement = transform.multiply(current_placement)
                        vertex_obj.Placement = new_placement

                        # Force refresh this specific object
                        vertex_obj.touch()
                        if hasattr(vertex_obj, 'ViewObject'):
                            vertex_obj.ViewObject.Visibility = False
                            vertex_obj.ViewObject.Visibility = True

                print(f"Transformed vertex group '{group_name}' with {len(vertex_group_obj.OutList)} vertices")
            else:
                print(f"Warning: Could not find vertex group '{group_name}' for transformation")

        # Force document recompute after all vertex transforms
        self.doc.recompute()

    @staticmethod
    def find_segments_with_transforms(doc=None):
        """Find all segments in document that have stored transforms."""
        if doc is None:
            doc = FreeCAD.ActiveDocument

        segments_with_transforms = []

        for obj in doc.Objects:
            if hasattr(obj, 'HasStoredTransform') and obj.HasStoredTransform:
                segments_with_transforms.append(obj)

        return segments_with_transforms

    @staticmethod
    def apply_transform_to_segment_and_vertices(segment_lcs_obj, new_transform):
        """Apply a new transform to a segment and its associated vertices (for external macros)."""
        try:
            # Get the segment object (find by LCS reference)
            segment_name = segment_lcs_obj.Label.replace('_lcs_base', '')
            segment_objects = [obj for obj in segment_lcs_obj.Document.Objects
                               if obj.Label.startswith(segment_name) and hasattr(obj, 'Shape')]

            # Apply transform to segment geometry
            for seg_obj in segment_objects:
                current_placement = seg_obj.Placement
                new_placement = new_transform.multiply(current_placement)
                seg_obj.Placement = new_placement

            # Apply to registered curve vertices
            for group_name in segment_lcs_obj.CurveVerticesGroups:
                curve_groups = segment_lcs_obj.Document.getObjectsByLabel(group_name)
                if curve_groups:
                    vertex_group_obj = curve_groups[0]
                    for vertex_obj in vertex_group_obj.OutList:
                        current_placement = vertex_obj.Placement
                        new_placement = new_transform.multiply(current_placement)
                        vertex_obj.Placement = new_placement

            # Update stored transform
            if hasattr(segment_lcs_obj, 'AppliedTransformMatrix'):
                old_transform = FlexSegment._parse_stored_transform(segment_lcs_obj)
                combined_transform = new_transform.multiply(old_transform) if old_transform else new_transform
                FlexSegment._store_transform_in_object(segment_lcs_obj, combined_transform)

            print(f"Applied external transform to segment {segment_name}")

        except Exception as e:
            print(f"Error applying external transform: {e}")

    @staticmethod
    def _parse_stored_transform(lcs_obj):
        """Helper to parse stored transform from FreeCAD object."""
        if hasattr(lcs_obj, 'AppliedTransformMatrix') and lcs_obj.AppliedTransformMatrix:
            try:
                matrix_values = [float(x) for x in lcs_obj.AppliedTransformMatrix.split(',')]
                matrix = FreeCAD.Matrix()
                for i in range(16):
                    matrix.A[i] = matrix_values[i]
                return FreeCAD.Placement(matrix)
            except:
                return None
        return None

    def store_pending_curve_vertices(self, group_name, curve_points):
        """Store curve points to create vertices AFTER transformation."""
        if not hasattr(self, 'pending_vertices'):
            self.pending_vertices = {}
        self.pending_vertices[group_name] = curve_points

    def move_to_top(self, transform):
        """Apply transform to reposition segment."""
        self.move_content(transform)

        # Store transform in FreeCAD properties for persistence
        self.store_applied_transform(transform)

        # NOW create vertices in their final transformed positions
        self._create_transformed_vertices(transform)

        # Call Python callbacks
        for callback in self.transform_callbacks:
            callback(transform)

    def _create_transformed_vertices(self, transform):
        """Create vertices directly in their final transformed positions."""
        if not hasattr(self, 'pending_vertices'):
            return

        for group_name, curve_points in self.pending_vertices.items():
            # Remove any existing group
            existing_groups = self.doc.getObjectsByLabel(group_name)
            for group in existing_groups:
                self.doc.removeObject(group.Name)

            # Create new group
            point_group = self.doc.addObject("App::DocumentObjectGroup", group_name)
            point_group.Label = group_name

            # Create vertices directly in final positions
            vertices = []
            for i, point in enumerate(curve_points):
                vertex_name = f"{group_name}_point_{i}"
                vertex_obj = self.doc.addObject('Part::Vertex', vertex_name)

                # Apply transform to original point to get final position
                original_placement = FreeCAD.Placement(FreeCAD.Vector(*point), FreeCAD.Rotation())
                final_placement = transform.multiply(original_placement)

                vertex_obj.Placement = final_placement
                vertices.append(vertex_obj)

            # Add to group and visualization
            point_group.addObjects(vertices)
            self.visualization_group.addObject(point_group)

            print(f"Created {len(vertices)} vertices in final positions for group '{group_name}'")

        # Clear pending vertices
        self.pending_vertices = {}

    @staticmethod
    def _store_transform_in_object(lcs_obj, transform):
        """Helper to store transform in FreeCAD object."""
        matrix = transform.toMatrix()
        matrix_str = ",".join([str(matrix.A[i]) for i in range(16)])
        lcs_obj.AppliedTransformMatrix = matrix_str
        lcs_obj.HasStoredTransform = True




