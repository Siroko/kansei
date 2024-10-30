/* eslint-disable */
const fs = require('fs');
const path = require('path');
const { fileURLToPath } = require('url');

const examplesDir = path.join(__dirname, '../dist/examples');
const distDir = path.join(__dirname, '../dist');

// Function to update import paths in HTML files
function updateImports(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');

    // Update import paths
    content = content.replace(/import\s+{([^}]+)}\s+from\s+['"](\.\.\/src\/[^'"]+)(\.ts)?['"]/g, (match, p1, p2) => {
        return `import {${p1}} from './${path.relative(path.dirname(filePath), path.join(distDir, p2.replace(/\/src\//, '/') + '.js')).replace(/\\/g, '/')}'`;
    });

    // Move index.html to the root of dist
    if (path.basename(filePath) === 'index.html') {
        const newFilePath = path.join(distDir, 'index.html');
        // Move the file to dist
        fs.renameSync(filePath, newFilePath);
    }

    // Write the updated content back to the file
    fs.writeFileSync(filePath, content, 'utf8');
}

// Read all HTML files in the examples directory
fs.readdirSync(examplesDir).forEach(file => {
    if (file.endsWith('.html')) {
        const filePath = path.join(examplesDir, file);
        updateImports(filePath);
    }
});

console.log('Imports updated successfully.');
