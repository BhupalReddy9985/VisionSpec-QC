function previewImage(event) {
    const reader = new FileReader();
    const preview = document.getElementById('image-preview');
    const dropText = document.getElementById('drop-text');
    const inspectBtn = document.getElementById('inspect-btn');
    
    reader.onload = function() {
        preview.src = reader.result;
        preview.style.display = 'block';
        dropText.style.display = 'none';
        inspectBtn.disabled = false;
        inspectBtn.style.opacity = '1';
    }
    
    if(event.target.files[0]) {
        reader.readAsDataURL(event.target.files[0]);
    }
}

async function runInspection() {
    const scanLine = document.getElementById('scanning-line');
    const fileInput = document.getElementById('file-input');
    const statusBadge = document.getElementById('status-badge');
    const confidenceVal = document.getElementById('confidence-val');
    const categoryVal = document.getElementById('category-val');
    const heatmapImg = document.getElementById('heatmap-img');
    const resultContent = document.getElementById('result-content');
    const noResult = document.getElementById('no-result');
    const inspectBtn = document.getElementById('inspect-btn');
    
    if(!fileInput.files[0]) return;
    
    // Toggle UI State
    scanLine.style.display = 'block';
    inspectBtn.innerText = 'INSPECTION IN PROGRESS...';
    inspectBtn.disabled = true;
    resultContent.style.opacity = '0.5';
    
    // Prepare Data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Hide Scanning Line after a short delay (visual effect)
        setTimeout(() => {
            scanLine.style.display = 'none';
            inspectBtn.disabled = false;
            inspectBtn.innerText = 'START INSPECTION';
            
            // Result Display
            noResult.style.display = 'none';
            resultContent.style.display = 'block';
            resultContent.style.opacity = '1';
            
            // Status Logic
            statusBadge.innerText = data.status;
            statusBadge.className = data.status === 'PASS' ? 'status-pill status-pass' : 'status-pill status-defect';
            
            confidenceVal.innerText = `${data.confidence}%`;
            categoryVal.innerText = data.status === 'PASS' ? 'Soldering Joint OK' : 'Solder Flaw / Heat Signature Fault';
            
            // Update Heatmap
            heatmapImg.src = data.heatmap_image;
        }, 1500);
        
    } catch (err) {
        console.error('Inspection failed:', err);
        inspectBtn.innerText = 'START INSPECTION';
        inspectBtn.disabled = false;
        alert('Inspection failed due to server error or connection.');
    }
}

async function loadSimulation() {
    // For simplicity, we just trigger /simulation if implemented, 
    // or pick a random file from we know is there (not very efficient without server support)
    // Using the /simulation endpoint we added to app.py
    const inspectBtn = document.getElementById('inspect-btn');
    const preview = document.getElementById('image-preview');
    const dropText = document.getElementById('drop-text');
    
    try {
        const res = await fetch('/simulation');
        const data = await res.json();
        
        // Set simulate preview
        preview.src = data.image;
        preview.style.display = 'block';
        dropText.style.display = 'none';
        inspectBtn.disabled = false;
        
        // Convert base64 to File object (needed by /predict)
        const blob = await (await fetch(data.image)).blob();
        const file = new File([blob], data.filename, {type: 'image/png'});
        
        // Manually trigger inspection after a quick look
        const container = new DataTransfer();
        container.items.add(file);
        document.getElementById('file-input').files = container.files;
        
        runInspection();
    } catch (err) {
        console.error('Simulation fetch failed:', err);
    }
}
