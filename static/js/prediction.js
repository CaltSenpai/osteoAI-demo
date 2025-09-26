let analysisStartTime;

const imageInput = document.getElementById('imageInput');
const predictBtn = document.getElementById('predictBtn');
const uploadForm = document.getElementById('uploadForm');
const loadingIndicator = document.getElementById('loadingIndicator');
const results = document.getElementById('results');
const imagePreview = document.getElementById('imagePreview');
const newAnalysisBtn = document.getElementById('newAnalysis');
const downloadBtn = document.getElementById('downloadReport');

// Image input change handler
imageInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            alert('⚠️ Please select a valid image file (JPG, PNG, JPEG)');
            this.value = '';
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            alert('⚠️ File size too large. Please select an image under 16MB.');
            this.value = '';
            return;
        }

        predictBtn.disabled = false;
        
        // Show image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.innerHTML = `
                <img src="${e.target.result}" alt="Medical Image Preview">
                <p><strong>File:</strong> ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)</p>
            `;
        };
        reader.readAsDataURL(file);
    } else {
        predictBtn.disabled = true;
        imagePreview.innerHTML = '';
    }
});

// Form submission handler
uploadForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    analysisStartTime = Date.now();
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    
    // Show loading state
    loadingIndicator.style.display = 'block';
    results.style.display = 'none';
    predictBtn.disabled = true;
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Calculate analysis time
            const analysisTime = ((Date.now() - analysisStartTime) / 1000).toFixed(1);
            
            // Display results
            document.getElementById('resultText').textContent = data.result;
            document.getElementById('confidenceText').textContent = data.confidence;
            document.getElementById('analysisTime').textContent = `${analysisTime} seconds`;
            
            // Style result based on diagnosis
            const resultElement = document.getElementById('resultText');
            if (data.result.includes('Detected')) {
                resultElement.style.color = '#e74c3c';
                resultElement.style.fontWeight = 'bold';
            } else {
                resultElement.style.color = '#27ae60';
                resultElement.style.fontWeight = 'bold';
            }
            
            results.style.display = 'block';
            
            // Store results for report generation
            window.lastAnalysisResult = {
                ...data,
                analysisTime: analysisTime,
                fileName: imageInput.files[0].name,
                timestamp: new Date().toLocaleString()
            };
            
        } else {
            alert('❌ Error: ' + data.error);
            predictBtn.disabled = false;
        }
    } catch (error) {
        alert('❌ Network Error: ' + error.message);
        predictBtn.disabled = false;
    } finally {
        loadingIndicator.style.display = 'none';
    }
});

// New analysis button handler
newAnalysisBtn.addEventListener('click', function() {
    uploadForm.reset();
    imagePreview.innerHTML = '';
    results.style.display = 'none';
    predictBtn.disabled = true;
    delete window.lastAnalysisResult;
});

// Download report button handler
downloadBtn.addEventListener('click', function() {
    if (!window.lastAnalysisResult) {
        alert('⚠️ No analysis results to download');
        return;
    }
    
    generateAndDownloadReport(window.lastAnalysisResult);
});

// Report generation function
function generateAndDownloadReport(data) {
    const reportContent = `
OSTEOAI ANALYSIS REPORT
=======================

Analysis Details:
- Date & Time: ${data.timestamp}
- File Name: ${data.fileName}
- Analysis Duration: ${data.analysisTime} seconds

Results:
- Diagnosis: ${data.result}
- Confidence Level: ${data.confidence}
- Raw Probability: ${(data.probability * 100).toFixed(2)}%

Important Disclaimer:
This AI analysis is for educational and research purposes only. 
It should not be used as a substitute for professional medical 
diagnosis. Always consult with qualified healthcare providers 
for proper medical evaluation and treatment.

Generated by OsteoAI - ${new Date().toLocaleString()}
    `.trim();
    
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `OsteoAI_Report_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Drag and drop functionality
const fileInputLabel = document.querySelector('.file-input-label');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    fileInputLabel.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    fileInputLabel.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    fileInputLabel.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    fileInputLabel.style.background = 'linear-gradient(135deg, #e9ecef, #dee2e6)';
    fileInputLabel.style.borderColor = '#667eea';
    fileInputLabel.style.transform = 'scale(1.02)';
}

function unhighlight(e) {
    fileInputLabel.style.background = 'linear-gradient(135deg, #f8f9fa, #e9ecef)';
    fileInputLabel.style.borderColor = '#bdc3c7';
    fileInputLabel.style.transform = 'scale(1)';
}

fileInputLabel.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        imageInput.files = files;
        imageInput.dispatchEvent(new Event('change'));
    }
}