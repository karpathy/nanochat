#!/usr/bin/env python3
"""
Generate an interactive 3D visualization of a phosphate molecule (PO₄³⁻).

This script creates a self-contained HTML file with Three.js that displays
a phosphate ion in tetrahedral geometry with interactive controls.
"""

import json
import math
import os


def generate_phosphate_html():
    """Generate HTML with Three.js for 3D phosphate molecule visualization."""
    
    # Calculate tetrahedral coordinates for oxygen atoms around phosphorus
    # Phosphorus at origin, oxygens at vertices of tetrahedron
    bond_length = 1.5  # Approximate P-O bond length in Angstroms
    
    # Tetrahedral geometry: vertices of a regular tetrahedron
    # Using standard tetrahedral coordinates
    tet_coords = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1)
    ]
    
    # Normalize to bond length
    oxygen_positions = []
    for x, y, z in tet_coords:
        length = math.sqrt(x*x + y*y + z*z)
        oxygen_positions.append({
            'x': x / length * bond_length,
            'y': y / length * bond_length,
            'z': z / length * bond_length
        })
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phosphate Molecule (PO₄³⁻) - 3D Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: hidden;
            height: 100vh;
        }}
        
        #container {{
            width: 100%;
            height: 100vh;
            position: relative;
        }}
        
        #info {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 300px;
            z-index: 100;
        }}
        
        h1 {{
            font-size: 24px;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .formula {{
            font-size: 20px;
            color: #667eea;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        
        .info-text {{
            font-size: 14px;
            color: #666;
            line-height: 1.6;
            margin-bottom: 10px;
        }}
        
        .legend {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 2px solid #eee;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 14px;
        }}
        
        .color-box {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid #333;
        }}
        
        .controls {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 2px solid #eee;
            font-size: 12px;
            color: #888;
        }}
        
        .controls p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h1>Phosphate Ion</h1>
        <div class="formula">PO₄³⁻</div>
        <div class="info-text">
            Tetrahedral molecular geometry with one phosphorus atom bonded to four oxygen atoms.
        </div>
        <div class="info-text">
            <strong>Bond Angle:</strong> ~109.5°<br>
            <strong>Geometry:</strong> Tetrahedral<br>
            <strong>Charge:</strong> -3
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="color-box" style="background: #ff6b35;"></div>
                <span>Phosphorus (P)</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #ff0000;"></div>
                <span>Oxygen (O)</span>
            </div>
        </div>
        <div class="controls">
            <p><strong>Controls:</strong></p>
            <p>🖱️ Left Click + Drag: Rotate</p>
            <p>🖱️ Right Click + Drag: Pan</p>
            <p>🖱️ Scroll: Zoom</p>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        camera.position.z = 6;
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        document.getElementById('container').appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        scene.add(directionalLight);
        
        const pointLight = new THREE.PointLight(0xffffff, 0.5);
        pointLight.position.set(-5, -5, 5);
        scene.add(pointLight);
        
        // Materials
        const phosphorusMaterial = new THREE.MeshPhongMaterial({{
            color: 0xff6b35,
            shininess: 100,
            specular: 0x444444
        }});
        
        const oxygenMaterial = new THREE.MeshPhongMaterial({{
            color: 0xff0000,
            shininess: 100,
            specular: 0x444444
        }});
        
        const bondMaterial = new THREE.MeshPhongMaterial({{
            color: 0x888888,
            shininess: 50
        }});
        
        // Create molecule group
        const molecule = new THREE.Group();
        
        // Phosphorus atom (center)
        const phosphorusGeometry = new THREE.SphereGeometry(0.4, 32, 32);
        const phosphorus = new THREE.Mesh(phosphorusGeometry, phosphorusMaterial);
        phosphorus.position.set(0, 0, 0);
        phosphorus.castShadow = true;
        molecule.add(phosphorus);
        
        // Oxygen atoms and bonds
        const oxygenPositions = {json.dumps(oxygen_positions)};
        
        oxygenPositions.forEach((pos, index) => {{
            // Oxygen atom
            const oxygenGeometry = new THREE.SphereGeometry(0.35, 32, 32);
            const oxygen = new THREE.Mesh(oxygenGeometry, oxygenMaterial);
            oxygen.position.set(pos.x, pos.y, pos.z);
            oxygen.castShadow = true;
            molecule.add(oxygen);
            
            // Bond (cylinder from P to O)
            const bondLength = Math.sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
            const bondGeometry = new THREE.CylinderGeometry(0.08, 0.08, bondLength, 16);
            const bond = new THREE.Mesh(bondGeometry, bondMaterial);
            
            // Position and orient the bond
            bond.position.set(pos.x / 2, pos.y / 2, pos.z / 2);
            
            // Calculate rotation to point from origin to oxygen
            const direction = new THREE.Vector3(pos.x, pos.y, pos.z).normalize();
            const axis = new THREE.Vector3(0, 1, 0);
            const quaternion = new THREE.Quaternion().setFromUnitVectors(axis, direction);
            bond.setRotationFromQuaternion(quaternion);
            
            molecule.add(bond);
        }});
        
        scene.add(molecule);
        
        // Mouse controls
        let isDragging = false;
        let isPanning = false;
        let previousMousePosition = {{ x: 0, y: 0 }};
        
        renderer.domElement.addEventListener('mousedown', (e) => {{
            if (e.button === 0) {{ // Left click
                isDragging = true;
            }} else if (e.button === 2) {{ // Right click
                isPanning = true;
            }}
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
        }});
        
        renderer.domElement.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;
                
                molecule.rotation.y += deltaX * 0.01;
                molecule.rotation.x += deltaY * 0.01;
            }} else if (isPanning) {{
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;
                
                camera.position.x -= deltaX * 0.01;
                camera.position.y += deltaY * 0.01;
            }}
            
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
        }});
        
        renderer.domElement.addEventListener('mouseup', () => {{
            isDragging = false;
            isPanning = false;
        }});
        
        renderer.domElement.addEventListener('contextmenu', (e) => {{
            e.preventDefault();
        }});
        
        // Zoom with mouse wheel
        renderer.domElement.addEventListener('wheel', (e) => {{
            e.preventDefault();
            camera.position.z += e.deltaY * 0.01;
            camera.position.z = Math.max(2, Math.min(15, camera.position.z));
        }});
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            
            // Gentle auto-rotation when not dragging
            if (!isDragging && !isPanning) {{
                molecule.rotation.y += 0.002;
            }}
            
            renderer.render(scene, camera);
        }}
        
        animate();
    </script>
</body>
</html>"""
    
    return html_content


def main():
    """Main function to generate the phosphate 3D visualization."""
    print("Generating 3D phosphate molecule visualization...")
    
    html_content = generate_phosphate_html()
    
    # Save to file
    output_file = os.path.join(os.path.dirname(__file__), 'phosphate_3d.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Generated: {output_file}")
    print(f"✓ Open the file in a web browser to view the 3D phosphate molecule")
    print(f"✓ Features: Interactive rotation, zoom, and pan controls")
    
    return output_file


if __name__ == '__main__':
    main()
