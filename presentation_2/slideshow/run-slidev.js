#!/usr/bin/env node
// Wrapper script to load localStorage polyfill before running slidev

// Load the polyfill first
require('./setup-localstorage.cjs');

// Now run slidev
const { spawn } = require('child_process');
const path = require('path');

// Get the slidev binary path
const slidevPath = require.resolve('@slidev/cli/bin/slidev');

// Get all arguments except the script name
const args = process.argv.slice(2);

// Spawn slidev with all arguments
const child = spawn('node', [slidevPath, ...args], {
  stdio: 'inherit',
  env: process.env
});

child.on('exit', (code) => {
  process.exit(code || 0);
});

