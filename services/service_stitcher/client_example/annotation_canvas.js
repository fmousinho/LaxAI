/**
 * Client-Side Annotation Manager for LaxAI Video Annotation
 * 
 * This module provides Canvas-based annotation rendering that matches
 * the server-side OpenCV rendering using the shared annotation schema.
 */

class AnnotationCanvas {
  constructor(canvasId, apiBaseUrl) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.apiBaseUrl = apiBaseUrl;
    this.sessionId = null;
    this.currentFrameId = null;
    this.currentRecipe = null;
    this.colorPalette = {};
    this.selectedDetection = null;
    
    // Bind event handlers
    this.canvas.addEventListener('click', this.handleCanvasClick.bind(this));
  }
  
  /**
   * Load video session
   */
  async loadVideo(tenantId, videoPath) {
    const response = await fetch(`${this.apiBaseUrl}/video/load`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tenant_id: tenantId, video_path: videoPath})
    });
    
    const data = await response.json();
    this.sessionId = data.session_id;
    return data;
  }
  
  /**
   * Load and display a specific frame
   */
  async loadFrame(frameId) {
    if (!this.sessionId) {
      throw new Error('No video session loaded');
    }
    
    // Fetch image and recipe in parallel
    const [imageBlob, recipeData] = await Promise.all([
      this.fetchFrameImage(frameId),
      this.fetchFrameRecipe(frameId)
    ]);
    
    // Load image
    const imageUrl = URL.createObjectURL(imageBlob);
    const img = new Image();
    
    return new Promise((resolve) => {
      img.onload = () => {
        // Set canvas size to match image
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        
        // Draw base image
        this.ctx.drawImage(img, 0, 0);
        
        // Store recipe and render annotations
        this.currentRecipe = recipeData.recipe;
        this.currentFrameId = frameId;
        this.renderAnnotations();
        
        // Clean up
        URL.revokeObjectURL(imageUrl);
        resolve();
      };
      img.src = imageUrl;
    });
  }
  
  /**
   * Fetch raw frame image
   */
  async fetchFrameImage(frameId, format = 'png') {
    const response = await fetch(
      `${this.apiBaseUrl}/video/frames/${this.sessionId}/${frameId}/image?format=${format}`
    );
    return await response.blob();
  }
  
  /**
   * Fetch annotation recipe
   */
  async fetchFrameRecipe(frameId) {
    const response = await fetch(
      `${this.apiBaseUrl}/video/frames/${this.sessionId}/${frameId}/recipe`
    );
    return await response.json();
  }
  
  /**
   * Render annotations from current recipe
   */
  renderAnnotations() {
    if (!this.currentRecipe) return;
    
    this.currentRecipe.instructions.forEach(instruction => {
      switch(instruction.type) {
        case 'bbox':
          this.drawBbox(instruction);
          break;
        case 'label':
          this.drawLabel(instruction);
          break;
        case 'point':
          this.drawPoint(instruction);
          break;
        case 'line':
          this.drawLine(instruction);
          break;
      }
    });
  }
  
  /**
   * Draw bounding box annotation
   */
  drawBbox(instruction) {
    const [x1, y1, x2, y2] = instruction.coords;
    const style = this.getStyle(instruction.style_preset, instruction.player_id);
    
    // Draw rectangle
    this.ctx.strokeStyle = style.bbox_color;
    this.ctx.lineWidth = style.bbox_thickness;
    this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    
    // Draw label
    const label = instruction.label_text || `P${instruction.player_id}`;
    const fullLabel = instruction.confidence 
      ? `${label} ${instruction.confidence.toFixed(2)}`
      : label;
    
    this.ctx.font = `${style.label_font_size}px Arial`;
    const textMetrics = this.ctx.measureText(fullLabel);
    
    // Label background
    this.ctx.fillStyle = style.label_bg_color;
    this.ctx.fillRect(x1, y1 - 20, textMetrics.width + 10, 20);
    
    // Label text
    this.ctx.fillStyle = style.label_text_color;
    this.ctx.fillText(fullLabel, x1 + 5, y1 - 5);
  }
  
  /**
   * Draw label annotation
   */
  drawLabel(instruction) {
    const [x, y] = instruction.coords;
    const style = this.getStyle(instruction.style_preset, instruction.player_id);
    const label = instruction.label_text || `P${instruction.player_id}`;
    
    this.ctx.font = `${style.label_font_size}px Arial`;
    const textMetrics = this.ctx.measureText(label);
    
    // Background
    this.ctx.fillStyle = style.label_bg_color;
    this.ctx.fillRect(x, y - 18, textMetrics.width + 8, 20);
    
    // Text
    this.ctx.fillStyle = style.label_text_color;
    this.ctx.fillText(label, x + 4, y - 3);
  }
  
  /**
   * Draw point marker
   */
  drawPoint(instruction) {
    const [x, y] = instruction.coords;
    const style = this.getStyle(instruction.style_preset, instruction.player_id);
    
    // Filled circle
    this.ctx.fillStyle = style.bbox_color;
    this.ctx.beginPath();
    this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
    this.ctx.fill();
    
    // Outline
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();
  }
  
  /**
   * Draw line
   */
  drawLine(instruction) {
    const [x1, y1, x2, y2] = instruction.coords;
    const style = this.getStyle(instruction.style_preset, instruction.player_id);
    
    this.ctx.strokeStyle = style.bbox_color;
    this.ctx.lineWidth = style.bbox_thickness;
    this.ctx.beginPath();
    this.ctx.moveTo(x1, y1);
    this.ctx.lineTo(x2, y2);
    this.ctx.stroke();
  }
  
  /**
   * Get style configuration for player and preset
   */
  getStyle(preset, playerId) {
    const baseColor = this.getPlayerColor(playerId);
    
    const styles = {
      default: {
        bbox_color: baseColor,
        bbox_thickness: 2,
        label_bg_color: 'black',
        label_text_color: 'white',
        label_font_size: 14
      },
      highlighted: {
        bbox_color: baseColor,
        bbox_thickness: 3,
        label_bg_color: baseColor,
        label_text_color: 'white',
        label_font_size: 16
      },
      dimmed: {
        bbox_color: this.dimColor(baseColor),
        bbox_thickness: 1,
        label_bg_color: 'rgba(0,0,0,0.5)',
        label_text_color: 'white',
        label_font_size: 12
      }
    };
    
    return styles[preset] || styles.default;
  }
  
  /**
   * Get color for player ID
   */
  getPlayerColor(playerId) {
    // Use cached color or fetch from server
    if (!this.colorPalette[playerId]) {
      // For now, use a simple hash-based color
      const hue = (playerId * 137.5) % 360;
      this.colorPalette[playerId] = `hsl(${hue}, 70%, 50%)`;
    }
    return this.colorPalette[playerId];
  }
  
  /**
   * Dim a color
   */
  dimColor(color) {
    // Simple dimming by reducing opacity
    return color.replace(')', ', 0.5)').replace('rgb', 'rgba').replace('hsl', 'hsla');
  }
  
  /**
   * Handle canvas click for detection selection
   */
  handleCanvasClick(event) {
    if (!this.currentRecipe) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Find clicked detection
    for (const instruction of this.currentRecipe.instructions) {
      if (instruction.type === 'bbox') {
        const [x1, y1, x2, y2] = instruction.coords;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
          this.selectDetection(instruction);
          break;
        }
      }
    }
  }
  
  /**
   * Select a detection for editing
   */
  selectDetection(instruction) {
    this.selectedDetection = instruction;
    
    // Trigger custom event
    const event = new CustomEvent('detectionSelected', {
      detail: {
        tracker_id: instruction.tracker_id,
        player_id: instruction.player_id,
        bbox: instruction.coords
      }
    });
    this.canvas.dispatchEvent(event);
  }
  
  /**
   * Update player ID for selected detection
   */
  updatePlayerMapping(trackerId, newPlayerId) {
    if (!this.currentRecipe) return false;
    
    let updated = false;
    this.currentRecipe.instructions.forEach(inst => {
      if (inst.tracker_id === trackerId) {
        inst.player_id = newPlayerId;
        updated = true;
      }
    });
    
    if (updated) {
      // Redraw canvas
      this.redrawFrame();
    }
    
    return updated;
  }
  
  /**
   * Redraw current frame with updated annotations
   */
  async redrawFrame() {
    if (this.currentFrameId !== null) {
      await this.loadFrame(this.currentFrameId);
    }
  }
  
  /**
   * Navigate to next frame
   */
  async nextFrame() {
    const response = await fetch(
      `${this.apiBaseUrl}/video/next-frame/${this.sessionId}`
    );
    const metadata = await response.json();
    await this.loadFrame(metadata.frame_id);
    return metadata;
  }
  
  /**
   * Navigate to previous frame
   */
  async previousFrame() {
    const response = await fetch(
      `${this.apiBaseUrl}/video/previous-frame/${this.sessionId}`
    );
    const metadata = await response.json();
    await this.loadFrame(metadata.frame_id);
    return metadata;
  }
  
  /**
   * Generate video with all edits
   */
  async generateVideo(frameRecipes, fps = 30) {
    const response = await fetch(
      `${this.apiBaseUrl}/video/${this.sessionId}/generate-video`,
      {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          frame_recipes: frameRecipes,
          output_format: 'mp4',
          fps: fps
        })
      }
    );
    
    return await response.json();
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AnnotationCanvas;
}
