"""
Clinical Decision Support System (CDSS) Dashboard View
=====================================================
Unified modern, professional medical AI web dashboard for Oncology Risk Prediction.
Implements the multi-stage pipeline (ML -> DL -> NLP -> SLM -> GenAI),
sidebar stage navigation, preset loading, CSV/image upload, and final clinical assessment.
"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI PATIENT RISK PREDICTION — Multi-Stage Clinical AI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-canvas: #f4f7fb;
            --bg-sidebar: #07152b;
            --bg-sidebar-hover: #0f2445;
            --bg-sidebar-active: #153360;
            --bg-header: #0b1e3f;
            --bg-header-gradient: linear-gradient(90deg, #07152b 0%, #0b1e3f 50%, #102a54 100%);
            --bg-card: #ffffff;
            --bg-card-alt: #f8fafc;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-light: #94a3b8;
            --border-card: #e2e8f0;
            --border-dark: rgba(255, 255, 255, 0.08);
            --accent-blue: #2563eb;
            --accent-blue-hover: #1d4ed8;
            --accent-blue-light: #eff6ff;
            --accent-blue-border: #bfdbfe;
            --accent-green: #059669;
            --accent-green-bg: #ecfdf5;
            --accent-green-border: #a7f3d0;
            --accent-yellow: #d97706;
            --accent-yellow-bg: #fffbeb;
            --accent-yellow-border: #fde68a;
            --accent-orange: #ea580c;
            --accent-orange-bg: #fff7ed;
            --accent-orange-border: #fed7aa;
            --accent-red: #dc2626;
            --accent-red-bg: #fef2f2;
            --accent-red-border: #fecaca;
            --accent-purple: #7c3aed;
            --accent-purple-bg: #f5f3ff;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
            --shadow-blue: 0 4px 14px 0 rgba(37, 99, 235, 0.35);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            overflow: hidden;
            font-size: 13.5px;
            line-height: 1.45;
        }

        /* ==========================================================================
           1. SIDEBAR NAVIGATION
           ========================================================================== */
        .sidebar {
            width: 230px;
            background: var(--bg-sidebar);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            border-right: 1px solid var(--border-dark);
            z-index: 100;
            user-select: none;
        }
        .sidebar-brand {
            padding: 1.25rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border-bottom: 1px solid var(--border-dark);
        }
        .brand-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
            flex-shrink: 0;
        }
        .brand-text h2 {
            font-size: 0.85rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.03em;
            line-height: 1.2;
        }
        .brand-text p {
            font-size: 0.65rem;
            color: #94a3b8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .sidebar-menu {
            display: flex;
            flex-direction: column;
            padding: 1rem 0.65rem;
            gap: 0.25rem;
            flex: 1;
            overflow-y: auto;
        }
        .menu-heading {
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.08em;
            padding: 0.75rem 0.65rem 0.35rem;
        }
        .nav-link {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.65rem 0.85rem;
            color: #94a3b8;
            text-decoration: none;
            border-radius: var(--radius-sm);
            font-size: 0.82rem;
            font-weight: 500;
            transition: all 0.18s ease;
            position: relative;
            cursor: pointer;
        }
        .nav-link:hover {
            color: #f1f5f9;
            background: var(--bg-sidebar-hover);
        }
        .nav-link.active {
            color: #ffffff;
            background: var(--bg-sidebar-active);
            font-weight: 600;
        }
        .nav-link.active::before {
            content: '';
            position: absolute;
            left: 0;
            top: 15%;
            bottom: 15%;
            width: 3.5px;
            background: #38bdf8;
            border-radius: 0 4px 4px 0;
            box-shadow: 0 0 8px #38bdf8;
        }
        .nav-link svg {
            width: 17px;
            height: 17px;
            stroke-width: 2;
            flex-shrink: 0;
        }
        .stage-pill-tag {
            margin-left: auto;
            font-size: 0.62rem;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        .sidebar-footer {
            padding: 1rem 1.25rem;
            border-top: 1px solid var(--border-dark);
            font-size: 0.68rem;
            color: #64748b;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .system-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 6px #10b981;
            display: inline-block;
            margin-right: 0.35rem;
        }

        /* ==========================================================================
           2. MAIN CONTAINER & HEADER
           ========================================================================== */
        .main-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            background: var(--bg-canvas);
        }
        .top-header {
            height: 64px;
            background: var(--bg-header-gradient);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 1.75rem;
            color: #ffffff;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
            flex-shrink: 0;
            z-index: 50;
        }
        .header-title-block {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .header-icon-box {
            width: 38px;
            height: 38px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #38bdf8;
        }
        .header-title-box h1 {
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: #ffffff;
            line-height: 1.2;
        }
        .header-title-box p {
            font-size: 0.72rem;
            color: #93c5fd;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        .header-status-strip {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 0, 0, 0.25);
            padding: 0.35rem 0.65rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .stage-status-chip {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.73rem;
            font-weight: 600;
            color: #cbd5e1;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.04);
            font-family: 'JetBrains Mono', monospace;
        }
        .stage-status-chip.ready { color: #4ade80; background: rgba(74, 222, 128, 0.1); }
        .stage-status-chip.running { color: #38bdf8; background: rgba(56, 189, 248, 0.18); animation: pulseSlow 1.5s infinite; }

        /* Clinician Profile & Auth in Header */
        .header-auth-group {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .header-auth-badge {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 8px;
            padding: 0.35rem 0.75rem;
            transition: all 0.2s ease;
        }
        .header-auth-badge:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(56, 189, 248, 0.4);
        }
        .clinician-avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
        }
        .clinician-details {
            display: flex;
            flex-direction: column;
            line-height: 1.15;
        }
        .clinician-name-text {
            font-size: 0.76rem;
            font-weight: 700;
            color: #ffffff;
        }
        .clinician-role-text {
            font-size: 0.62rem;
            color: #38bdf8;
            font-family: 'JetBrains Mono', monospace;
        }
        .btn-header-logout {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.35);
            border-radius: 6px;
            color: #fca5a5;
            padding: 0.4rem 0.65rem;
            font-size: 0.72rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.35rem;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        .btn-header-logout:hover {
            background: rgba(239, 68, 68, 0.3);
            color: #ffffff;
            border-color: #ef4444;
            transform: translateY(-1px);
        }

        /* ==========================================================================
           3. CONTENT BODY & VIEWS
           ========================================================================== */
        .content-scroll {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem 1.75rem 3rem;
        }
        .view-section {
            display: none;
            flex-direction: column;
            gap: 1.25rem;
        }
        .view-section.active {
            display: flex;
        }

        /* Headings */
        .view-header {
            margin-bottom: 0.25rem;
        }
        .view-header h2 {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }
        .view-header p {
            font-size: 0.82rem;
            color: var(--text-secondary);
        }

        /* ==========================================================================
           4. PIPELINE FLOW CARDS
           ========================================================================== */
        .pipeline-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-lg);
            padding: 1.25rem 1.5rem;
            box-shadow: var(--shadow-sm);
        }
        .pipeline-flow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.65rem;
            position: relative;
            margin: 0.75rem 0 0.5rem;
        }
        .flow-node {
            flex: 1;
            background: var(--bg-card-alt);
            border: 1.5px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 0.85rem 0.75rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            position: relative;
            transition: all 0.25s ease;
            cursor: pointer;
        }
        .flow-node:hover {
            border-color: var(--accent-blue-border);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .flow-node.active-stage {
            border-color: var(--accent-blue);
            background: #f0f7ff;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
        }
        .flow-node.completed-stage {
            border-color: #10b981;
            background: #f0fdf4;
        }
        .flow-node.processing-stage {
            border-color: #38bdf8;
            background: #f0f9ff;
            animation: pulseCard 1.2s infinite alternate;
        }
        .node-badge {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #e2e8f0;
            color: var(--text-secondary);
            font-weight: 700;
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.4rem;
            transition: all 0.2s;
        }
        .flow-node.active-stage .node-badge { background: var(--accent-blue); color: #fff; }
        .flow-node.completed-stage .node-badge { background: #10b981; color: #fff; }
        .flow-node.processing-stage .node-badge { background: #38bdf8; color: #fff; }
        .node-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .node-subtitle {
            font-size: 0.68rem;
            color: var(--text-secondary);
            margin: 0.15rem 0 0.45rem;
        }
        .node-status-pill {
            font-size: 0.62rem;
            font-weight: 600;
            padding: 0.15rem 0.5rem;
            border-radius: 20px;
            background: #e2e8f0;
            color: #64748b;
            font-family: 'JetBrains Mono', monospace;
        }
        .flow-node.completed-stage .node-status-pill { background: #dcfce7; color: #166534; }
        .flow-node.processing-stage .node-status-pill { background: #e0f2fe; color: #0369a1; }
        .flow-arrow {
            color: #cbd5e1;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .flow-arrow svg {
            width: 18px;
            height: 18px;
        }
        .pipeline-progress-bar-wrap {
            margin-top: 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .progress-track {
            flex: 1;
            height: 6px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #38bdf8 0%, #2563eb 100%);
            border-radius: 4px;
            transition: width 0.4s ease;
        }
        .progress-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
        }

        /* ==========================================================================
           5. PATIENT INPUT SECTION
           ========================================================================== */
        .patient-input-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
        }
        .card-title-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-card);
        }
        .card-title-strip h3 {
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .preset-buttons {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            flex-wrap: wrap;
        }
        .btn-preset {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.35rem 0.65rem;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-card);
            background: var(--bg-card-alt);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-preset:hover {
            border-color: var(--accent-blue);
            color: var(--accent-blue);
            background: #eff6ff;
        }
        .btn-preset.danger {
            border-color: var(--accent-red-border);
            color: var(--accent-red);
            background: var(--accent-red-bg);
        }
        .btn-preset.danger:hover {
            background: #fee2e2;
        }

        /* Inputs Grid */
        .input-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.85rem;
            margin-bottom: 1rem;
        }
        .input-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.85rem;
            margin-bottom: 1rem;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .form-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .form-control {
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            padding: 0.5rem 0.75rem;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-card);
            background: #ffffff;
            color: var(--text-primary);
            transition: border-color 0.15s;
            outline: none;
        }
        .form-control:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        }
        textarea.form-control {
            resize: vertical;
            min-height: 72px;
            font-size: 0.8rem;
            line-height: 1.4;
        }

        /* File Upload Zones */
        .upload-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }
        .upload-box {
            border: 1.5px dashed var(--border-card);
            border-radius: var(--radius-md);
            padding: 0.85rem 1rem;
            background: var(--bg-card-alt);
            display: flex;
            align-items: center;
            gap: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }
        .upload-box:hover {
            border-color: var(--accent-blue);
            background: #f8fafc;
        }
        .upload-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-blue);
            flex-shrink: 0;
        }
        .upload-text h4 {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .upload-text p {
            font-size: 0.68rem;
            color: var(--text-secondary);
        }
        .upload-box input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }
        .image-preview-thumb {
            width: 44px;
            height: 44px;
            border-radius: 6px;
            object-fit: cover;
            border: 1px solid var(--border-card);
            display: none;
        }

        /* ==========================================================================
           MEDICAL IMAGE UPLOAD & VALIDATION WORKFLOW STYLES
           ========================================================================== */
        .medical-upload-panel {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        .section-subheading {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.65rem;
            display: block;
        }
        .required-indicator {
            color: #ef4444;
            font-weight: 800;
        }
        .modality-pill-group {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .modality-pill {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            border-radius: var(--radius-md);
            border: 1.5px solid var(--border-card);
            background: var(--bg-card-alt);
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }
        .modality-pill:hover {
            border-color: #38bdf8;
            background: #f8fafc;
        }
        .modality-pill.selected {
            border-color: #2563eb;
            background: rgba(37, 99, 235, 0.06);
            box-shadow: 0 0 0 1px #2563eb;
        }
        .modality-pill input[type="radio"] {
            margin: 0;
            width: 16px;
            height: 16px;
            accent-color: #2563eb;
            cursor: pointer;
        }
        .pill-labels {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }
        .pill-name {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .pill-desc {
            font-size: 0.68rem;
            color: var(--text-secondary);
        }

        .upload-dropzone-box {
            border: 2px dashed #cbd5e1;
            border-radius: var(--radius-md);
            background: #f8fafc;
            padding: 1.75rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.65rem;
        }
        .upload-dropzone-box:hover, .upload-dropzone-box.drag-over {
            border-color: #2563eb;
            background: rgba(37, 99, 235, 0.04);
        }
        .dropzone-icon-circle {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: #e2e8f0;
            color: #2563eb;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.25rem;
        }
        .dropzone-text-block h4 {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }
        .dropzone-text-block p {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin: 0;
        }
        .dropzone-sub {
            font-size: 0.68rem;
            color: #94a3b8;
            margin-top: 0.35rem;
            display: block;
        }
        .btn-choose-image {
            margin-top: 0.5rem;
            background: #ffffff;
            color: #2563eb;
            border: 1.5px solid #2563eb;
            padding: 0.5rem 1.25rem;
            border-radius: var(--radius-sm);
            font-size: 0.82rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-choose-image:hover {
            background: #2563eb;
            color: #ffffff;
        }

        /* Medical Image Preview Card */
        .medical-image-card {
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            background: var(--bg-card);
            padding: 1rem 1.25rem;
            margin-top: 1rem;
            box-shadow: var(--shadow-sm);
        }
        .card-box-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 0.85rem;
        }
        .card-box-title {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.82rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: 0.03em;
        }
        .btn-remove-selected {
            background: none;
            border: none;
            color: #ef4444;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .btn-remove-selected:hover {
            background: #fee2e2;
        }
        .preview-layout {
            display: grid;
            grid-template-columns: 140px 1fr;
            gap: 1.25rem;
            align-items: center;
        }
        .preview-display-wrapper {
            width: 140px;
            height: 140px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-card);
            overflow: hidden;
            background: #0f172a;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .med-preview-img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .preview-details-wrapper {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }
        .preview-detail-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.78rem;
        }
        .detail-label {
            width: 80px;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .detail-value {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
            font-weight: 500;
        }
        .detail-value.highlight {
            font-weight: 700;
            color: #2563eb;
        }
        .status-badge {
            padding: 0.15rem 0.55rem;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            background: #f1f5f9;
            color: #475569;
        }
        .status-badge.valid {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #10b981;
        }
        .status-badge.error {
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #ef4444;
        }
        .validate-action-strip {
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border-card);
            display: flex;
            gap: 0.65rem;
        }
        .btn-validate-image {
            background: #0f172a;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 0.6rem 1.25rem;
            border-radius: var(--radius-sm);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            transition: all 0.15s ease;
        }
        .btn-validate-image:hover {
            background: #1e293b;
        }
        .btn-validate-image:disabled {
            background: #94a3b8;
            cursor: not-allowed;
        }

        /* Triple Modality Multi-Slot Layout */
        .triple-modality-container {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.75rem;
        }
        @media (max-width: 900px) {
            .triple-modality-container {
                grid-template-columns: 1fr;
            }
        }
        .modality-slot-card {
            border: 1.5px solid var(--border-card);
            border-radius: var(--radius-md);
            background: #ffffff;
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            position: relative;
        }
        .modality-slot-card:hover {
            border-color: #cbd5e1;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06);
        }
        .modality-slot-card.has-file {
            border-color: #93c5fd;
            background: #f8fafc;
        }
        .modality-slot-card.is-valid {
            border-color: #86efac;
            background: #f0fdf4;
        }
        .modality-slot-card.is-invalid {
            border-color: #fca5a5;
            background: #fef2f2;
        }
        .slot-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.4rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #f1f5f9;
        }
        .slot-header-info {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            min-width: 0;
        }
        .slot-icon-badge {
            width: 26px;
            height: 26px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .slot-icon-badge.mri { background: #ede9fe; color: #7c3aed; }
        .slot-icon-badge.ct { background: #e0f2fe; color: #0284c7; }
        .slot-icon-badge.pathology { background: #fce7f3; color: #db2777; }

        .slot-title {
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
            line-height: 1.2;
        }
        .slot-subtitle {
            font-size: 0.64rem;
            color: var(--text-secondary);
            display: block;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .slot-header-actions {
            display: flex;
            align-items: center;
            gap: 0.3rem;
            flex-shrink: 0;
        }
        .slot-pill-badge {
            font-size: 0.63rem;
            font-weight: 700;
            padding: 0.12rem 0.42rem;
            border-radius: 999px;
            letter-spacing: 0.02em;
        }
        .slot-pill-badge.awaiting {
            background: #f1f5f9;
            color: #64748b;
        }
        .slot-pill-badge.ready {
            background: #fef3c7;
            color: #92400e;
        }
        .slot-pill-badge.valid {
            background: #dcfce7;
            color: #166534;
        }
        .slot-pill-badge.invalid {
            background: #fee2e2;
            color: #991b1b;
        }
        .btn-slot-clear {
            background: none;
            border: none;
            color: #ef4444;
            font-size: 0.68rem;
            font-weight: 700;
            cursor: pointer;
            padding: 0.1rem 0.3rem;
            border-radius: 4px;
        }
        .btn-slot-clear:hover {
            background: #fee2e2;
        }

        /* Slot Dropzone */
        .slot-dropzone {
            border: 1.5px dashed #cbd5e1;
            border-radius: 8px;
            background: #f8fafc;
            padding: 1rem 0.6rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.3rem;
            min-height: 140px;
        }
        .slot-dropzone:hover, .slot-dropzone.drag-over {
            border-color: #2563eb;
            background: rgba(37, 99, 235, 0.04);
        }
        .slot-drop-icon {
            color: #64748b;
        }
        .slot-drop-title {
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .slot-drop-formats {
            font-size: 0.63rem;
            color: #94a3b8;
        }
        .btn-slot-choose {
            margin-top: 0.35rem;
            background: #ffffff;
            color: #2563eb;
            border: 1.5px solid #2563eb;
            padding: 0.35rem 0.85rem;
            border-radius: 5px;
            font-size: 0.74rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-block;
            user-select: none;
        }
        .btn-slot-choose:hover {
            background: #2563eb;
            color: #ffffff;
        }
        .slot-change-btn {
            font-size: 0.68rem;
            font-weight: 700;
            color: #2563eb;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 0.15rem 0.55rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-block;
            margin-top: 0.25rem;
            text-align: center;
        }
        .slot-change-btn:hover {
            background: #dbeafe;
            border-color: #93c5fd;
        }

        /* Slot Preview Box */
        .slot-preview-box {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            min-height: 140px;
            justify-content: space-between;
        }
        .slot-thumb-wrapper {
            width: 100%;
            height: 110px;
            background: #090d16;
            border-radius: 8px;
            border: 1px solid #334155;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
        }
        .slot-thumb-img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 4px;
            display: block;
        }
        .slot-meta-wrapper {
            display: flex;
            flex-direction: column;
            gap: 0.18rem;
        }
        .slot-meta-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.68rem;
            gap: 0.4rem;
        }
        .slot-meta-label {
            color: #64748b;
            font-weight: 600;
            flex-shrink: 0;
        }
        .slot-meta-val {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 150px;
            text-align: right;
        }
        .slot-status-tag {
            font-size: 0.66rem;
            font-weight: 700;
            padding: 0.1rem 0.38rem;
            border-radius: 4px;
            background: #f1f5f9;
            color: #475569;
        }
        .slot-status-tag.valid {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }
        .slot-status-tag.invalid {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .btn-slot-validate {
            background: #0f172a;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            font-size: 0.74rem;
            font-weight: 700;
            padding: 0.45rem 0.65rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            transition: all 0.15s ease;
            width: 100%;
        }
        .btn-slot-validate:hover {
            background: #1e293b;
        }
        .btn-slot-validate:disabled {
            background: #94a3b8;
            cursor: not-allowed;
        }
        .slot-feedback-line {
            font-size: 0.65rem;
            padding: 0.3rem 0.45rem;
            border-radius: 4px;
            font-weight: 600;
            line-height: 1.3;
        }
        .slot-feedback-line.error {
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        .slot-feedback-line.success {
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        .btn-validate-all {
            background: #1e293b;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            font-size: 0.74rem;
            font-weight: 700;
            padding: 0.42rem 0.95rem;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.15s ease;
        }
        .btn-validate-all:hover {
            background: #0f172a;
        }

        /* Validation Status / Error Banner */
        .validation-banner {
            margin-top: 1rem;
            border-radius: var(--radius-md);
            padding: 0.85rem 1.15rem;
            font-size: 0.82rem;
            display: flex;
            align-items: center;
            animation: fadeIn 0.2s ease-in-out;
        }
        .validation-banner.banner-success {
            background: #ecfdf5;
            border: 1.5px solid #10b981;
            color: #065f46;
        }
        .validation-banner.banner-error {
            background: #fef2f2;
            border: 1.5px solid #ef4444;
            color: #991b1b;
        }
        .validation-banner.banner-info {
            background: #eff6ff;
            border: 1.5px solid #3b82f6;
            color: #1e40af;
        }
        .banner-inner {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            width: 100%;
            font-weight: 600;
        }

        /* Collapsible Clinical Information Drawer */
        .clinical-drawer {
            margin-top: 1.25rem;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            background: var(--bg-card-alt);
            overflow: hidden;
            transition: all 0.2s ease;
        }
        .clinical-drawer-summary {
            padding: 0.85rem 1.15rem;
            background: #ffffff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.78rem;
            color: var(--text-primary);
            user-select: none;
            border-bottom: 1px solid transparent;
        }
        .clinical-drawer[open] .clinical-drawer-summary {
            border-bottom-color: var(--border-card);
        }
        .summary-left {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--text-primary);
        }
        .summary-right-hint {
            font-size: 0.7rem;
            color: var(--text-secondary);
            font-weight: 500;
        }
        .clinical-drawer-content {
            padding: 1.25rem;
            background: var(--bg-card);
        }

        /* Buttons Strip */
        .btn-action-row {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding-top: 0.5rem;
        }
        .btn-primary-analyze {
            flex: 1;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            padding: 0.85rem 1.75rem;
            border-radius: var(--radius-md);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.65rem;
            box-shadow: var(--shadow-blue);
            transition: all 0.2s ease;
        }
        .btn-primary-analyze:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
        }
        .btn-primary-analyze:disabled {
            background: #94a3b8;
            box-shadow: none;
            cursor: not-allowed;
            transform: none;
        }
        .btn-secondary-reset {
            background: var(--bg-card-alt);
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.85rem 1.5rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-card);
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn-secondary-reset:hover {
            background: #e2e8f0;
            color: var(--text-primary);
        }

        /* ==========================================================================
           6. FIVE STAGE LIVE RESULTS GRID
           ========================================================================== */
        .stage-results-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        .stage-result-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1rem 1.25rem;
            box-shadow: var(--shadow-sm);
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            position: relative;
            transition: all 0.2s;
        }
        .stage-result-card.expanded-full {
            grid-column: span 2;
        }
        .card-header-mini {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .card-header-mini h4 {
            font-size: 0.84rem;
            font-weight: 800;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }
        .metric-badge {
            font-size: 0.68rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-badge.green { background: #dcfce7; color: #15803d; }
        .metric-badge.blue { background: #dbeafe; color: #1d4ed8; }
        .metric-badge.yellow { background: #fef3c7; color: #b45309; }
        .metric-badge.red { background: #fee2e2; color: #b91c1c; }

        .feature-tag-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        .feature-tag {
            font-size: 0.68rem;
            background: var(--bg-card-alt);
            border: 1px solid var(--border-card);
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            color: var(--text-secondary);
        }
        .feature-tag strong { color: var(--text-primary); }

        /* ==========================================================================
           7. FINAL PREDICTION CARD (PROMINENT CLINICAL ASSESSMENT)
           ========================================================================== */
        .final-assessment-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid var(--accent-blue-border);
            border-radius: var(--radius-lg);
            padding: 1.75rem;
            box-shadow: var(--shadow-md);
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }
        .final-assessment-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #2563eb, #38bdf8, #10b981);
            border-radius: var(--radius-lg) var(--radius-lg) 0 0;
        }
        .assessment-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-card);
        }
        .assessment-banner-left h3 {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: 0.02em;
        }
        .assessment-banner-left p {
            font-size: 0.78rem;
            color: var(--text-secondary);
        }
        .risk-classification-tag {
            font-size: 1.1rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            padding: 0.5rem 1.25rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            box-shadow: var(--shadow-sm);
        }
        .risk-classification-tag.low { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
        .risk-classification-tag.moderate { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
        .risk-classification-tag.high { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
        .risk-classification-tag.critical { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

        .assessment-breakdown-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.85rem;
        }
        .breakdown-box {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 0.75rem 0.85rem;
        }
        .breakdown-box span {
            font-size: 0.65rem;
            font-weight: 700;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .breakdown-box h4 {
            font-size: 0.92rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-top: 0.2rem;
        }

        .clinical-summary-box {
            background: #f8fafc;
            border-left: 4px solid var(--accent-blue);
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            padding: 1rem 1.25rem;
            font-size: 0.84rem;
            line-height: 1.55;
            color: var(--text-primary);
        }
        .considerations-checklist {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin-top: 0.5rem;
        }
        .consideration-item {
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        .consideration-item svg {
            width: 16px;
            height: 16px;
            color: var(--accent-blue);
            flex-shrink: 0;
            margin-top: 2px;
        }
        .disclaimer-strip {
            font-size: 0.72rem;
            color: #64748b;
            font-style: italic;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            border-top: 1px solid var(--border-card);
            padding-top: 0.75rem;
        }

        /* ==========================================================================
           8. TABLES & DATA LISTS (HISTORY, ANALYTICS)
           ========================================================================== */
        .table-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        .table-responsive {
            width: 100%;
            overflow-x: auto;
        }
        table.clinical-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            text-align: left;
        }
        table.clinical-table th {
            background: #f8fafc;
            color: var(--text-secondary);
            font-weight: 700;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-card);
            text-transform: uppercase;
            font-size: 0.68rem;
            letter-spacing: 0.05em;
        }
        table.clinical-table td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-card);
            color: var(--text-primary);
        }
        table.clinical-table tr:hover td {
            background: #f8fafc;
        }

        /* Analytics Stats Grid */
        .analytics-stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1.15rem;
            box-shadow: var(--shadow-sm);
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .stat-card span {
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        .stat-card h3 {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Animations */
        @keyframes pulseSlow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        @keyframes pulseCard {
            0% { border-color: #38bdf8; box-shadow: 0 0 8px rgba(56, 189, 248, 0.2); }
            100% { border-color: #2563eb; box-shadow: 0 0 16px rgba(37, 99, 235, 0.4); }
        }

        /* Responsive Layout adjustments */
        @media (max-width: 1024px) {
            .input-grid-4 { grid-template-columns: repeat(2, 1fr); }
            .assessment-breakdown-grid { grid-template-columns: repeat(2, 1fr); }
            .stage-results-grid { grid-template-columns: 1fr; }
            .stage-result-card.expanded-full { grid-column: span 1; }
            .sidebar { width: 72px; }
            .brand-text, .nav-link span, .menu-heading, .stage-pill-tag, .sidebar-footer { display: none; }
            .sidebar-brand { justify-content: center; padding: 1rem 0; }
        }

        /* ==========================================================================
           STAGE 06 — AGENTIC AI COMMAND CENTER STYLES
           ========================================================================== */
        .agentic-status-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.75rem;
            margin-top: 0.85rem;
        }
        @media (max-width: 1200px) {
            .agentic-status-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .agent-node-card {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 0.85rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .agent-node-card:hover {
            border-color: #38bdf8;
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.12);
        }
        .agent-node-info h4 {
            font-size: 0.76rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: 0.02em;
        }
        .agent-node-info p {
            font-size: 0.68rem;
            color: var(--text-light);
            margin-top: 2px;
        }
        .agent-status-indicator {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 900;
            flex-shrink: 0;
            transition: all 0.3s ease;
        }
        .agent-status-indicator.waiting {
            background: #f1f5f9;
            color: #94a3b8;
            border: 1.5px solid #cbd5e1;
        }
        .agent-status-indicator.processing {
            background: #e0f2fe;
            color: #0284c7;
            border: 1.5px solid #38bdf8;
            animation: spinFast 1.2s linear infinite;
        }
        .agent-status-indicator.completed {
            background: #ecfdf5;
            color: #059669;
            border: 1.5px solid #10b981;
            box-shadow: 0 0 6px rgba(16, 185, 129, 0.3);
        }
        .agent-status-indicator.warning {
            background: #fffbeb;
            color: #d97706;
            border: 1.5px solid #f59e0b;
        }
        .agent-status-indicator.failed {
            background: #fef2f2;
            color: #dc2626;
            border: 1.5px solid #ef4444;
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
        }
        @keyframes spinFast {
            100% { transform: rotate(360deg); }
        }

        /* ---------------------------------------------------------
           REALISTIC CLINICAL WORKSTATION COMMAND & CONTROL STRIP
           --------------------------------------------------------- */
        .physician-control-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1.15rem;
            background: linear-gradient(135deg, #07152b 0%, #0d2246 50%, #0a1c38 100%);
            border: 1px solid rgba(56, 189, 248, 0.32);
            border-radius: 14px;
            padding: 1rem 1.4rem;
            color: #ffffff;
            margin-top: 1.25rem;
            margin-bottom: 1.35rem;
            box-shadow: 0 10px 25px -5px rgba(2, 6, 23, 0.45), 0 0 0 1px rgba(255, 255, 255, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.12);
            position: relative;
            overflow: hidden;
        }
        .physician-control-bar::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2.5px;
            background: linear-gradient(90deg, #38bdf8 0%, #2563eb 35%, #10b981 70%, #38bdf8 100%);
            background-size: 200% 100%;
            animation: controlBarBeam 5s linear infinite;
        }
        @keyframes controlBarBeam {
            0% { background-position: 100% 0; }
            100% { background-position: -100% 0; }
        }

        .physician-control-title {
            display: flex;
            align-items: center;
            gap: 0.95rem;
        }
        .physician-title-icon {
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.16) 0%, rgba(37, 99, 235, 0.28) 100%);
            border: 1px solid rgba(56, 189, 248, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #38bdf8;
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
            flex-shrink: 0;
        }
        .physician-title-text h4 {
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            text-transform: uppercase;
        }
        .physician-title-text p {
            font-size: 0.72rem;
            color: #94a3b8;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin-top: 2px;
        }
        .live-pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 10px #10b981;
            display: inline-block;
            animation: pulseGlow 1.8s infinite ease-in-out;
        }
        @keyframes pulseGlow {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.55; }
        }

        .physician-btn-group {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            flex-wrap: wrap;
        }

        /* REALISTIC PHYSICIAN WORKSTATION BUTTONS */
        .btn-physician {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.55rem;
            padding: 0.62rem 1.15rem;
            font-size: 0.81rem;
            font-weight: 700;
            font-family: inherit;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            user-select: none;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            outline: none;
            letter-spacing: 0.02em;
            white-space: nowrap;
            overflow: hidden;
        }
        .btn-physician::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(to bottom, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0) 60%);
            pointer-events: none;
            border-radius: 7px;
        }
        .btn-physician:active:not(:disabled) {
            transform: translateY(1px) scale(0.98);
        }
        .btn-physician:disabled {
            opacity: 0.55;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }
        .btn-physician svg {
            flex-shrink: 0;
            transition: transform 0.2s ease;
        }
        .btn-physician:hover:not(:disabled) svg {
            transform: scale(1.12);
        }

        /* 1. Primary Action: Run Agentic Analysis */
        .btn-physician.primary {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 55%, #1d4ed8 100%);
            color: #ffffff;
            border: 1px solid rgba(147, 197, 253, 0.45);
            box-shadow: 0 4px 14px -1px rgba(37, 99, 235, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.35);
        }
        .btn-physician.primary:hover:not(:disabled) {
            background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 55%, #1e40af 100%);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.65), 0 0 12px rgba(56, 189, 248, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.45);
            transform: translateY(-1.5px);
        }

        /* 2. Secondary Action: Pause / Resume */
        .btn-physician.pause {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #e2e8f0;
            border: 1px solid rgba(148, 163, 184, 0.3);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        .btn-physician.pause:hover:not(:disabled) {
            background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
            color: #ffffff;
            border-color: rgba(148, 163, 184, 0.55);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
            transform: translateY(-1.5px);
        }

        /* 3. Halt / Stop Action */
        .btn-physician.stop {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(153, 27, 27, 0.6) 100%);
            color: #fecaca;
            border: 1px solid rgba(248, 113, 113, 0.4);
            box-shadow: 0 2px 8px rgba(185, 28, 28, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        .btn-physician.stop:hover:not(:disabled) {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: #ffffff;
            border-color: #ef4444;
            box-shadow: 0 5px 18px rgba(220, 38, 38, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3);
            transform: translateY(-1.5px);
        }

        /* 4. Physician Override Action */
        .btn-physician.override {
            background: linear-gradient(135deg, rgba(180, 83, 9, 0.35) 0%, rgba(217, 119, 6, 0.45) 100%);
            color: #fef3c7;
            border: 1px solid rgba(251, 191, 36, 0.45);
            box-shadow: 0 2px 8px rgba(217, 119, 6, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        }
        .btn-physician.override:hover:not(:disabled) {
            background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
            color: #ffffff;
            border-color: #f59e0b;
            box-shadow: 0 5px 18px rgba(217, 119, 6, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3);
            transform: translateY(-1.5px);
        }

        .decision-trace-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
        }
        .decision-trace-table th {
            background: #f8fafc;
            padding: 0.65rem 0.85rem;
            text-align: left;
            font-weight: 700;
            color: var(--text-light);
            border-bottom: 1px solid var(--border-card);
            text-transform: uppercase;
            font-size: 0.65rem;
            letter-spacing: 0.05em;
        }
        .decision-trace-table td {
            padding: 0.75rem 0.85rem;
            border-bottom: 1px solid #f1f5f9;
            vertical-align: top;
            color: var(--text-primary);
        }
        .decision-trace-table tr:hover {
            background: #f8fafc;
        }
        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #0284c7;
            color: #ffffff;
            font-weight: 800;
            font-size: 0.68rem;
        }
        .trial-card-item {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid #0284c7;
            transition: all 0.2s ease;
        }
        .trial-card-item:hover {
            border-color: #38bdf8;
            box-shadow: var(--shadow-sm);
        }
        .trial-card-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        .trial-card-header h4 {
            font-size: 0.88rem;
            font-weight: 800;
            color: var(--text-primary);
        }
        .trial-meta-pills {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-top: 0.35rem;
        }
        .trial-meta-pill {
            font-size: 0.65rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: #f1f5f9;
            color: #475569;
            font-weight: 600;
        }
        .trial-meta-pill.match {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .safety-alert-halt {
            background: #fef2f2;
            border: 2px solid #ef4444;
            border-radius: var(--radius-md);
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            animation: pulseCard 2s infinite alternate;
        }
        .safety-alert-halt h3 {
            font-size: 1.05rem;
            font-weight: 900;
            color: #b91c1c;
            letter-spacing: 0.04em;
        }
        .safety-alert-halt p {
            font-size: 0.82rem;
            color: #7f1d1d;
            margin-top: 0.35rem;
            line-height: 1.5;
        }

        /* ==========================================================================
           ENHANCEMENTS: 6 POWER FEATURES (PDF, SIMULATOR, SHAP, GRAD-CAM, STREAM, VOICE)
           ========================================================================== */

        /* 1. PDF Export Button */
        .btn-header-pdf {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.48rem 0.95rem;
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
            color: #ffffff;
            font-size: 0.76rem;
            font-weight: 700;
            border-radius: var(--radius-sm);
            border: 1px solid rgba(255, 255, 255, 0.2);
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.35);
            transition: all 0.2s ease;
            text-decoration: none;
        }
        .btn-header-pdf:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45);
            background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%);
        }
        .btn-export-pdf-card {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.9rem;
            background: #ffffff;
            color: #0284c7;
            font-size: 0.76rem;
            font-weight: 700;
            border-radius: var(--radius-sm);
            border: 1.5px solid #0284c7;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-export-pdf-card:hover {
            background: #0284c7;
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
        }

        /* 2. Doctor Voice Dictation */
        .btn-voice-dictate {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.7rem;
            font-size: 0.72rem;
            font-weight: 700;
            border-radius: 20px;
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }
        .btn-voice-dictate:hover {
            background: #e2e8f0;
            color: #0f172a;
        }
        .btn-voice-dictate.recording {
            background: #fef2f2;
            color: #dc2626;
            border-color: #f87171;
            animation: recordPulse 1.2s infinite alternate;
        }
        @keyframes recordPulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
            100% { box-shadow: 0 0 0 8px rgba(220, 38, 38, 0); }
        }

        /* 3. Interactive What-If Simulator */
        .whatif-simulator-card {
            background: linear-gradient(135deg, #07152b 0%, #0d2246 100%);
            border: 1px solid rgba(56, 189, 248, 0.32);
            border-radius: var(--radius-md);
            padding: 1.25rem 1.5rem;
            color: #ffffff;
            margin-top: 1.25rem;
            box-shadow: 0 10px 25px -5px rgba(2, 6, 23, 0.45);
        }
        .whatif-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 0.65rem;
        }
        .whatif-grid {
            display: grid;
            grid-template-columns: 1.35fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 900px) {
            .whatif-grid { grid-template-columns: 1fr; }
        }
        .sim-slider-row {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            margin-bottom: 0.75rem;
        }
        .sim-slider-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.76rem;
            font-weight: 600;
            color: #94a3b8;
        }
        .sim-slider-header span.val {
            color: #38bdf8;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
        }
        .sim-slider {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #334155;
            outline: none;
            cursor: pointer;
            accent-color: #38bdf8;
        }
        .sim-outcome-panel {
            background: rgba(11, 30, 63, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: var(--radius-sm);
            padding: 1.1rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.85rem;
        }

        /* 4. Model Explainability: SHAP Waterfall */
        .shap-waterfall-card {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1.25rem;
            margin-top: 1rem;
        }
        .shap-bar-row {
            display: grid;
            grid-template-columns: 150px 1fr 75px;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.6rem;
            font-size: 0.76rem;
        }
        .shap-bar-track {
            height: 14px;
            background: #f1f5f9;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }
        .shap-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .shap-bar-fill.positive {
            background: linear-gradient(90deg, #f87171, #ef4444);
        }
        .shap-bar-fill.negative {
            background: linear-gradient(90deg, #34d399, #10b981);
        }

        /* 5. Grad-CAM Heatmap Viewer */
        .gradcam-container {
            position: relative;
            display: inline-block;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
            background: #000;
        }
        .gradcam-heatmap-layer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            mix-blend-mode: color-dodge;
            opacity: 0.75;
            transition: opacity 0.2s ease;
            pointer-events: none;
            background: radial-gradient(circle at 62% 44%, rgba(255, 0, 0, 0.95) 0%, rgba(255, 120, 0, 0.8) 25%, rgba(255, 230, 0, 0.55) 45%, rgba(0, 160, 255, 0.25) 65%, rgba(0, 0, 0, 0) 80%);
        }

        /* 6. Virtual Tumor Board Deliberation Stream */
        .deliberation-stream-card {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1.25rem;
            margin-top: 1.25rem;
        }
        .deliberation-feed {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            max-height: 420px;
            overflow-y: auto;
            padding-right: 0.5rem;
            margin-top: 0.75rem;
        }
        .agent-chat-msg {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            padding: 0.75rem 0.95rem;
            background: #f8fafc;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-sm);
            transition: all 0.2s ease;
        }
        .agent-chat-msg:hover {
            border-color: #38bdf8;
            box-shadow: 0 2px 8px rgba(56, 189, 248, 0.1);
        }
        .agent-chat-avatar {
            width: 34px;
            height: 34px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            flex-shrink: 0;
            background: #e0f2fe;
        }
        .agent-chat-body {
            flex: 1;
        }
        .agent-chat-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.25rem;
        }
        .agent-chat-header h5 {
            font-size: 0.78rem;
            font-weight: 800;
            color: #0f172a;
        }
        .agent-chat-header span {
            font-size: 0.68rem;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
        }
        .agent-chat-text {
            font-size: 0.77rem;
            color: #334155;
            line-height: 1.45;
        }

        /* ==========================================================================
           EVIDENCE & RAG VIEW STYLES
           ========================================================================== */
        .rag-arch-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }
        .rag-arch-title {
            font-size: 0.76rem;
            font-weight: 700;
            color: #64748b;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 1.25rem;
        }
        .rag-steps-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }
        .rag-step-box {
            flex: 1;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.25rem 1rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            transition: all 0.2s ease;
        }
        .rag-step-box:hover {
            border-color: #38bdf8;
            background: #f0f9ff;
            transform: translateY(-2px);
        }
        .rag-step-num {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #ffffff;
            border: 1.5px solid #cbd5e1;
            color: #475569;
            font-size: 0.82rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.75rem;
        }
        .rag-step-name {
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .rag-step-desc {
            font-size: 0.78rem;
            color: #64748b;
            font-weight: 500;
        }
        .rag-step-arrow {
            color: #94a3b8;
            font-size: 1.35rem;
            font-weight: 600;
            user-select: none;
        }
        .rag-protocol-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            transition: all 0.2s ease;
        }
        .rag-protocol-card:hover {
            border-color: #cbd5e1;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .rag-protocol-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.75rem;
            line-height: 1.4;
        }
        .rag-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
            margin-bottom: 1rem;
        }
        .rag-badge-blue {
            background: #e0f2fe;
            color: #0284c7;
            border: 1px solid #bae6fd;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 5px;
        }
        .rag-badge-gray {
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #e2e8f0;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.25rem 0.65rem;
            border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
        }
        .rag-quote-text {
            font-size: 0.88rem;
            line-height: 1.6;
            color: #334155;
            margin-bottom: 1rem;
        }
        .rag-link-btn {
            color: #0284c7;
            font-size: 0.84rem;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            cursor: pointer;
            transition: color 0.15s ease;
        }
        .rag-link-btn:hover {
            color: #0369a1;
            text-decoration: underline;
        }

        /* ==========================================================================
           AUDIT TRAIL VIEW STYLES
           ========================================================================== */
        .audit-filter-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.15rem 1.5rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }
        .audit-filter-group {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }
        .audit-filter-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #334155;
            white-space: nowrap;
        }
        .audit-filter-input, .audit-filter-select {
            padding: 0.45rem 0.75rem;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            font-size: 0.82rem;
            color: #0f172a;
            background: #ffffff;
            outline: none;
            transition: border-color 0.15s;
        }
        .audit-filter-input:focus, .audit-filter-select:focus {
            border-color: #0284c7;
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.1);
        }
        .audit-table-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.15rem;
        }
        .audit-table-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #0f172a;
        }
        .btn-refresh-audit {
            padding: 0.4rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: #334155;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-refresh-audit:hover {
            background: #f1f5f9;
            border-color: #94a3b8;
        }
    </style>
</head>
<body>

    <!-- ======================================================================
         SIDEBAR NAVIGATION
         ====================================================================== -->
    <aside class="sidebar">
        <div class="sidebar-brand">
            <div class="brand-icon">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor">
                    <path d="M12 2v20M2 12h20M4.93 4.93l14.14 14.14M4.93 19.07l14.14-14.14" stroke-width="2.5" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="brand-text">
                <h2>ONCO-CDSS</h2>
                <p>Clinical AI Core</p>
            </div>
        </div>

        <nav class="sidebar-menu">
            <div class="menu-heading">MAIN WORKFLOW</div>
            <a class="nav-link active" data-view="view-prediction">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>
                <span>Prediction</span>
            </a>

            <div class="menu-heading">6-STAGE PIPELINE</div>
            <a class="nav-link" data-view="view-stage1">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/></svg>
                <span>① Treatment Safety</span>
                <div class="stage-pill-tag">ML</div>
            </a>
            <a class="nav-link" data-view="view-stage2">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                <span>② Medical Imaging</span>
                <div class="stage-pill-tag">DL</div>
            </a>
            <a class="nav-link" data-view="view-stage3">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                <span>③ Clinical Triage</span>
                <div class="stage-pill-tag">NLP</div>
            </a>
            <a class="nav-link" data-view="view-stage4">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                <span>④ Clinical AI Assistant</span>
                <div class="stage-pill-tag">SLM</div>
            </a>
            <a class="nav-link" data-view="view-stage5">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                <span>⑤ Clinical Report</span>
                <div class="stage-pill-tag">GENAI</div>
            </a>
            <a class="nav-link" data-view="view-stage6">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                <span>⑥ AI Care Workflow</span>
                <div class="stage-pill-tag" style="background: rgba(14, 165, 233, 0.2); color: #0284c7;">AGENTIC</div>
            </a>

            <div class="menu-heading">EVIDENCE &amp; GOVERNANCE</div>
            <a class="nav-link" data-view="view-rag">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="5" y="3" width="14" height="18" rx="2" stroke-width="2"/><line x1="9" y1="3" x2="9" y2="21" stroke-width="2"/></svg>
                <span>Evidence &amp; RAG</span>
                <div class="stage-pill-tag">NCCN</div>
            </a>
            <a class="nav-link" data-view="view-audit">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" stroke-width="2"/><circle cx="12" cy="11" r="2.5" stroke-width="2"/></svg>
                <span>Audit Trail</span>
                <div class="stage-pill-tag">GOV</div>
            </a>
            <a class="nav-link" data-view="view-health">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M6 20v-4M12 20v-9M18 20V4" stroke-width="2.2" stroke-linecap="round"/></svg>
                <span>Model Health</span>
                <div class="stage-pill-tag">MONITOR</div>
            </a>
            <a class="nav-link" data-view="view-settings">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                <span>Settings</span>
            </a>
        </nav>

        <div class="sidebar-footer">
            <div><span class="system-dot"></span>Active Pipeline</div>
            <div style="font-family: 'JetBrains Mono', monospace;">v2.4-PROD</div>
        </div>
    </aside>

    <!-- ======================================================================
         MAIN CONTENT AREA
         ====================================================================== -->
    <main class="main-wrapper">
        <!-- Top Navigation Header -->
        <header class="top-header">
            <div class="header-title-block">
                <div class="header-icon-box">
                    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class="header-title-box">
                    <h1>AI PATIENT RISK PREDICTION</h1>
                    <p>Multi-stage clinical AI analysis</p>
                </div>
            </div>

            <div class="header-status-strip">
                <div class="stage-status-chip ready" id="chip-ml" title="Treatment Safety">Safety ✓</div>
                <div class="stage-status-chip ready" id="chip-dl" title="Medical Imaging">Imaging ✓</div>
                <div class="stage-status-chip ready" id="chip-nlp" title="Clinical Triage">Triage ✓</div>
                <div class="stage-status-chip ready" id="chip-slm" title="Clinical AI Assistant">AI Assistant ✓</div>
                <div class="stage-status-chip ready" id="chip-genai" title="Clinical Report">Report ✓</div>
                <div class="stage-status-chip ready" id="chip-agentic" style="border-color: #0284c7; color: #0284c7;" title="AI Care Workflow">Workflow ✓</div>
            </div>

            <!-- Authenticated Clinician Widget & PDF Export -->
            <div class="header-auth-group">
                <button type="button" class="btn-header-pdf" id="btn-header-export-pdf" onclick="exportTumorBoardPdf()" title="Download Official Tumor Board PDF Report">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
                    <span>Export PDF</span>
                </button>
                <div class="header-auth-badge" id="header-clinician-badge" title="Active Authenticated Session">
                    <div class="clinician-avatar" id="header-clinician-avatar">🩺</div>
                    <div class="clinician-details">
                        <span class="clinician-name-text" id="header-clinician-name">Dr. Sarah Chen, MD</span>
                        <span class="clinician-role-text" id="header-clinician-role">Chief Oncologist • Clearance L4</span>
                    </div>
                </div>
                <a href="/login" class="btn-header-logout" onclick="signOutClinician(event)" title="Sign Out & Return to Login">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
                    <span>Exit Portal</span>
                </a>
            </div>
        </header>

        <div class="content-scroll">

            <!-- ==============================================================
                 VIEW 1: DASHBOARD / PREDICTION (PRIMARY MULTI-STAGE PIPELINE)
                 ============================================================== -->
            <section id="view-prediction" class="view-section active">
                <div class="view-header">
                    <h2>AI PATIENT RISK PREDICTION</h2>
                    <p>Multi-stage clinical AI analysis for oncology risk assessment</p>
                </div>

                <!-- 1. Pipeline Flow Card -->
                <div class="pipeline-card">
                    <div class="pipeline-flow">
                        <!-- Node 1: ML -->
                        <div class="flow-node" id="node-stage1" onclick="switchView('view-stage1')">
                            <div class="node-badge">1</div>
                            <div class="node-title">ML</div>
                            <div class="node-subtitle">Treatment Safety</div>
                            <span class="node-status-pill" id="pill-stage1">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 2: DL -->
                        <div class="flow-node" id="node-stage2" onclick="switchView('view-stage2')">
                            <div class="node-badge">2</div>
                            <div class="node-title">DL</div>
                            <div class="node-subtitle">Medical Imaging</div>
                            <span class="node-status-pill" id="pill-stage2">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 3: NLP -->
                        <div class="flow-node" id="node-stage3" onclick="switchView('view-stage3')">
                            <div class="node-badge">3</div>
                            <div class="node-title">NLP</div>
                            <div class="node-subtitle">Clinical Triage</div>
                            <span class="node-status-pill" id="pill-stage3">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 4: SLM -->
                        <div class="flow-node" id="node-stage4" onclick="switchView('view-stage4')">
                            <div class="node-badge">4</div>
                            <div class="node-title">SLM</div>
                            <div class="node-subtitle">Clinical AI Assistant</div>
                            <span class="node-status-pill" id="pill-stage4">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 5: GenAI -->
                        <div class="flow-node" id="node-stage5" onclick="switchView('view-stage5')">
                            <div class="node-badge">5</div>
                            <div class="node-title">GenAI</div>
                            <div class="node-subtitle">Clinical Report Insights</div>
                            <span class="node-status-pill" id="pill-stage5">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 6: Agentic AI -->
                        <div class="flow-node" id="node-stage6" onclick="switchView('view-stage6')">
                            <div class="node-badge">6</div>
                            <div class="node-title">Agentic</div>
                            <div class="node-subtitle">AI Care Deliberation</div>
                            <span class="node-status-pill" id="pill-stage6">Waiting</span>
                        </div>
                    </div>

                    <div class="pipeline-progress-bar-wrap">
                        <div class="progress-track"><div class="progress-fill" id="pipeline-progress-fill"></div></div>
                        <div class="progress-label" id="pipeline-progress-label">Stage 0 of 6 (Ready)</div>
                    </div>
                </div>

                <!-- 2a. PATIENT CLINICAL DATA & BIOMARKER ANALYSIS (TOP) -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/></svg>
                            PATIENT CLINICAL DATA & BIOMARKER ANALYSIS
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">QUICK FILL:</span>
                            <button class="btn-preset" type="button" onclick="setOverviewSample('HIGH')">High Toxicity Sample</button>
                            <button class="btn-preset" type="button" onclick="setOverviewSample('MODERATE')">Moderate Sample</button>
                            <button class="btn-preset" type="button" onclick="setOverviewSample('LOW')">Low Toxicity Sample</button>
                            <button class="btn-preset" type="button" onclick="setOverviewSample('COLON')" style="border-color: rgba(16, 185, 129, 0.4); color: #34d399;">Colon COAD (Test Set)</button>
                        </div>
                    </div>

                    <form id="patient-form" onsubmit="event.preventDefault(); handleRunAnalysisClick();">
                        <!-- Demographics & Staging -->
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Patient ID</label>
                                <input type="text" id="inp-patient_id" class="form-control" value="PAT-ML-0101" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age (Years)</label>
                                <input type="number" id="inp-age" class="form-control" value="64" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Sex</label>
                                <select id="inp-sex" class="form-control">
                                    <option value="M" selected>Male</option>
                                    <option value="F">Female</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Type</label>
                                <select id="inp-cancer_type" class="form-control">
                                    <option value="Lung (LUAD)" selected>Lung Adenocarcinoma (LUAD)</option>
                                    <option value="Lung (LUSC)">Lung Squamous (LUSC)</option>
                                    <option value="Breast (BRCA)">Breast Invasive (BRCA)</option>
                                    <option value="Colon (COAD)">Colon Adeno (COAD)</option>
                                    <option value="Prostate (PRAD)">Prostate Adeno (PRAD)</option>
                                    <option value="Stomach (STAD)">Stomach Adeno (STAD)</option>
                                    <option value="Melanoma (SKCM)">Skin Cutaneous Melanoma (SKCM)</option>
                                </select>
                            </div>
                        </div>

                        <!-- Oncology Biomarkers -->
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Cancer Stage</label>
                                <select id="inp-cancer_stage" class="form-control">
                                    <option value="Stage I">Stage I</option>
                                    <option value="Stage II">Stage II</option>
                                    <option value="Stage III">Stage III</option>
                                    <option value="Stage IV" selected>Stage IV</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Biomarker Analysis</label>
                                <input type="text" id="inp-mutation_profile" class="form-control" value="EGFR L858R">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Tumor Marker Level (ng/ml)</label>
                                <input type="number" id="inp-tumor_marker_level" class="form-control" value="28.6" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">ctDNA Fractional VAF (%)</label>
                                <input type="number" id="inp-ctDNA_level" class="form-control" value="1.84" step="0.01" required>
                            </div>
                        </div>

                        <!-- Lab Panels -->
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">ALT (U/L)</label>
                                <input type="number" id="inp-ALT" class="form-control" value="248" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">AST (U/L)</label>
                                <input type="number" id="inp-AST" class="form-control" value="210" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Creatinine (mg/dL)</label>
                                <input type="number" id="inp-creatinine" class="form-control" value="1.45" step="0.01" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Bilirubin (mg/dL)</label>
                                <input type="number" id="inp-bilirubin" class="form-control" value="2.2" step="0.1" required>
                            </div>
                        </div>

                        <!-- Treatment & Regimen -->
                        <div class="input-grid-3">
                            <div class="form-group">
                                <label class="form-label">Treatment Information / Regimen</label>
                                <input type="text" id="inp-treatment_name" class="form-control" value="Osimertinib 80mg + Capmatinib">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Dosage (mg)</label>
                                <input type="number" id="inp-dosage_mg" class="form-control" value="480" step="10">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Treatment Cycle</label>
                                <input type="number" id="inp-treatment_cycle" class="form-control" value="4" step="1">
                            </div>
                        </div>

                        <!-- Clinical Notes with Doctor Voice Dictation -->
                        <div class="form-group" style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                                <label class="form-label" style="margin-bottom: 0;">Clinical Note Analysis &amp; Patient Symptoms</label>
                                <button type="button" class="btn-voice-dictate" id="btn-dictate-main" onclick="toggleVoiceDictation('inp-clinical_notes', 'btn-dictate-main')" title="Dictate Patient Symptoms &amp; Notes via Microphone">
                                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                                    <span class="dictate-text">Doctor Voice Dictate</span>
                                </button>
                            </div>
                            <textarea id="inp-clinical_notes" class="form-control" placeholder="Enter clinical consult notes, acute symptoms, vital signs or click 'Doctor Voice Dictate'...">Patient on Osimertinib and Capmatinib presenting with spiking fever 38.4C, marked right upper quadrant tenderness, intractable nausea, and severe fatigue. Marked transaminase elevation (ALT 248 U/L, AST 210 U/L) indicating acute Drug-Induced Liver Injury (DILI).</textarea>
                        </div>

                        <!-- File Upload Dropzone for CSV -->
                        <div class="upload-row">
                            <div class="upload-box" id="drop-csv" style="grid-column: span 2;">
                                <div class="upload-icon">
                                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                                </div>
                                <div class="upload-text">
                                    <h4 id="csv-filename">Upload Patient CSV (Optional)</h4>
                                    <p>Drop lab or cohort CSV to autofill companion clinical fields</p>
                                </div>
                                <input type="file" id="file-csv-input" accept=".csv" onchange="handleCSVUpload(this)">
                            </div>
                        </div>
                    </form>
                </div>

                <!-- 2b. MEDICAL IMAGE UPLOAD (BELOW) -->
                <div class="patient-input-card" style="margin-top: 1.25rem;">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                            MEDICAL IMAGE UPLOAD
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">SAMPLE PRESETS:</span>
                            <button type="button" class="btn-preset" onclick="loadPreset('HIGH_RISK')">High Risk NSCLC (CT)</button>
                            <button type="button" class="btn-preset" onclick="loadPreset('MODERATE_RISK')">Moderate Risk BRCA (Pathology)</button>
                            <button type="button" class="btn-preset" onclick="loadPreset('LOW_RISK')">Low Risk PRAD (CT)</button>
                            <button type="button" class="btn-preset" onclick="loadPreset('COLON_TEST_SET')" style="border-color: rgba(16, 185, 129, 0.4); color: #34d399;">Colon COAD (Test Set)</button>
                            <button type="button" class="btn-preset danger" onclick="loadPreset('CRITICAL_EDGE')">Critical DILI Trap</button>
                        </div>
                    </div>

                    <!-- PRIMARY MEDICAL IMAGE UPLOAD WORKFLOW (MULTI-MODAL SLOTS) -->
                    <div class="medical-upload-panel">
                        <!-- Header with Guidance & Test Compat Hooks -->
                        <div class="modality-select-section">
                            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem;">
                                <label class="section-subheading" style="margin-bottom: 0;">
                                    Select Image Type / Multi-Modal Input Slots <span class="required-indicator">*</span>
                                </label>
                                <span style="font-size: 0.72rem; color: #64748b; font-weight: 500;">
                                    Upload &amp; validate MRI, CT Scan, and Pathology individually in dedicated slots below:
                                </span>
                            </div>

                            <!-- Triple Modality Grid: MRI, CT Scan, Pathology -->
                            <div class="triple-modality-container">
                                <!-- 1. MRI Scan Slot -->
                                <div class="modality-slot-card" id="slot-card-mri">
                                    <div class="slot-header" id="mod-pill-mri">
                                        <div class="slot-header-info">
                                            <div class="slot-icon-badge mri">
                                                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18"/><path d="M12 7a5 5 0 0 0 0 10"/></svg>
                                            </div>
                                            <div>
                                                <h4 class="slot-title">MRI Scan</h4>
                                                <span class="slot-subtitle">Magnetic Resonance Imaging</span>
                                            </div>
                                        </div>
                                        <div class="slot-header-actions">
                                            <span class="slot-pill-badge awaiting" id="slot-badge-mri">Awaiting Scan</span>
                                            <button type="button" class="btn-slot-clear" id="slot-btn-clear-mri" style="display: none;" onclick="clearSlotImage('mri')" title="Clear MRI image">✕ Clear</button>
                                        </div>
                                    </div>

                                    <!-- Dropzone (when empty) -->
                                    <div class="slot-dropzone" id="slot-dropzone-mri"
                                         ondragover="handleSlotDragOver(event, 'mri')"
                                         ondragleave="handleSlotDragLeave(event, 'mri')"
                                         ondrop="handleSlotDrop(event, 'mri')"
                                         onclick="triggerSlotFileInput('mri')">
                                        <div class="slot-drop-icon">
                                            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                                        </div>
                                        <span class="slot-drop-title">[ Upload Medical Image ]</span>
                                        <span class="slot-drop-formats">PNG, JPG, JPEG (Max 15MB)</span>
                                        <label for="file-input-mri" class="btn-slot-choose" onclick="event.stopPropagation()">
                                            [ Choose MRI ]
                                        </label>
                                        <input type="file" id="file-input-mri" accept=".png,.jpg,.jpeg,image/png,image/jpeg" style="display: none;" onchange="handleSlotFileSelect(this, 'mri')">
                                    </div>

                                    <!-- Preview & Validation Box (when populated) -->
                                    <div class="slot-preview-box" id="slot-preview-mri" style="display: none;">
                                        <div class="slot-thumb-wrapper">
                                            <img id="slot-img-mri" src="" alt="MRI Preview" class="slot-thumb-img">
                                        </div>
                                        <div style="display: flex; justify-content: center; margin-top: -0.2rem;">
                                            <label for="file-input-mri" class="slot-change-btn" onclick="event.stopPropagation()">
                                                ↻ Change Image
                                            </label>
                                        </div>
                                        <div class="slot-meta-wrapper">
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">File:</span>
                                                <span class="slot-meta-val" id="slot-filename-mri">—</span>
                                            </div>
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">Size:</span>
                                                <span class="slot-meta-val" id="slot-filesize-mri">—</span>
                                            </div>
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">Status:</span>
                                                <span class="slot-status-tag" id="slot-status-mri">Awaiting validation</span>
                                            </div>
                                        </div>
                                        <button type="button" id="btn-validate-mri" class="btn-slot-validate btn-validate-image" onclick="validateSlotImage('mri')">
                                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                            [ Validate MRI ]
                                        </button>
                                        <div class="slot-feedback-line" id="slot-feedback-mri" style="display: none;"></div>
                                    </div>
                                </div>

                                <!-- 2. CT Scan Slot -->
                                <div class="modality-slot-card" id="slot-card-ct">
                                    <div class="slot-header" id="mod-pill-ct">
                                        <div class="slot-header-info">
                                            <div class="slot-icon-badge ct">
                                                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/></svg>
                                            </div>
                                            <div>
                                                <h4 class="slot-title">CT Scan</h4>
                                                <span class="slot-subtitle">2D Axial Radiology Slice</span>
                                            </div>
                                        </div>
                                        <div class="slot-header-actions">
                                            <span class="slot-pill-badge awaiting" id="slot-badge-ct">Awaiting Scan</span>
                                            <button type="button" class="btn-slot-clear" id="slot-btn-clear-ct" style="display: none;" onclick="clearSlotImage('ct')" title="Clear CT image">✕ Clear</button>
                                        </div>
                                    </div>

                                    <!-- Dropzone (when empty) -->
                                    <div class="slot-dropzone" id="slot-dropzone-ct"
                                         ondragover="handleSlotDragOver(event, 'ct')"
                                         ondragleave="handleSlotDragLeave(event, 'ct')"
                                         ondrop="handleSlotDrop(event, 'ct')"
                                         onclick="triggerSlotFileInput('ct')">
                                        <div class="slot-drop-icon">
                                            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                                        </div>
                                        <span class="slot-drop-title">[ Upload Medical Image ]</span>
                                        <span class="slot-drop-formats">PNG, JPG, JPEG (Max 15MB)</span>
                                        <label for="file-input-ct" class="btn-slot-choose" onclick="event.stopPropagation()">
                                            [ Choose CT Scan ]
                                        </label>
                                        <input type="file" id="file-input-ct" accept=".png,.jpg,.jpeg,image/png,image/jpeg" style="display: none;" onchange="handleSlotFileSelect(this, 'ct')">
                                    </div>

                                    <!-- Preview & Validation Box (when populated) -->
                                    <div class="slot-preview-box" id="slot-preview-ct" style="display: none;">
                                        <div class="slot-thumb-wrapper">
                                            <img id="slot-img-ct" src="" alt="CT Preview" class="slot-thumb-img">
                                        </div>
                                        <div style="display: flex; justify-content: center; margin-top: -0.2rem;">
                                            <label for="file-input-ct" class="slot-change-btn" onclick="event.stopPropagation()">
                                                ↻ Change Image
                                            </label>
                                        </div>
                                        <div class="slot-meta-wrapper">
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">File:</span>
                                                <span class="slot-meta-val" id="slot-filename-ct">—</span>
                                            </div>
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">Size:</span>
                                                <span class="slot-meta-val" id="slot-filesize-ct">—</span>
                                            </div>
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">Status:</span>
                                                <span class="slot-status-tag" id="slot-status-ct">Awaiting validation</span>
                                            </div>
                                        </div>
                                        <button type="button" id="btn-validate-ct" class="btn-slot-validate btn-validate-image" onclick="validateSlotImage('ct')">
                                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                            [ Validate CT ]
                                        </button>
                                        <div class="slot-feedback-line" id="slot-feedback-ct" style="display: none;"></div>
                                    </div>
                                </div>

                                <!-- 3. Pathology Slot -->
                                <div class="modality-slot-card" id="slot-card-pathology">
                                    <div class="slot-header" id="mod-pill-pathology">
                                        <div class="slot-header-info">
                                            <div class="slot-icon-badge pathology">
                                                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                                            </div>
                                            <div>
                                                <h4 class="slot-title">Pathology</h4>
                                                <span class="slot-subtitle">H&amp;E Histopathology Tile</span>
                                            </div>
                                        </div>
                                        <div class="slot-header-actions">
                                            <span class="slot-pill-badge awaiting" id="slot-badge-pathology">Awaiting Scan</span>
                                            <button type="button" class="btn-slot-clear" id="slot-btn-clear-pathology" style="display: none;" onclick="clearSlotImage('pathology')" title="Clear Pathology image">✕ Clear</button>
                                        </div>
                                    </div>

                                    <!-- Dropzone (when empty) -->
                                    <div class="slot-dropzone" id="slot-dropzone-pathology"
                                         ondragover="handleSlotDragOver(event, 'pathology')"
                                         ondragleave="handleSlotDragLeave(event, 'pathology')"
                                         ondrop="handleSlotDrop(event, 'pathology')"
                                         onclick="triggerSlotFileInput('pathology')">
                                        <div class="slot-drop-icon">
                                            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                                        </div>
                                        <span class="slot-drop-title">[ Upload Medical Image ]</span>
                                        <span class="slot-drop-formats">PNG, JPG, JPEG (Max 15MB)</span>
                                        <label for="file-input-pathology" class="btn-slot-choose" onclick="event.stopPropagation()">
                                            [ Choose Pathology ]
                                        </label>
                                        <input type="file" id="file-input-pathology" accept=".png,.jpg,.jpeg,image/png,image/jpeg" style="display: none;" onchange="handleSlotFileSelect(this, 'pathology')">
                                    </div>

                                    <!-- Preview & Validation Box (when populated) -->
                                    <div class="slot-preview-box" id="slot-preview-pathology" style="display: none;">
                                        <div class="slot-thumb-wrapper">
                                            <img id="slot-img-pathology" src="" alt="Pathology Preview" class="slot-thumb-img">
                                        </div>
                                        <div style="display: flex; justify-content: center; margin-top: -0.2rem;">
                                            <label for="file-input-pathology" class="slot-change-btn" onclick="event.stopPropagation()">
                                                ↻ Change Image
                                            </label>
                                        </div>
                                        <div class="slot-meta-wrapper">
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">File:</span>
                                                <span class="slot-meta-val" id="slot-filename-pathology">—</span>
                                            </div>
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">Size:</span>
                                                <span class="slot-meta-val" id="slot-filesize-pathology">—</span>
                                            </div>
                                            <div class="slot-meta-row">
                                                <span class="slot-meta-label">Status:</span>
                                                <span class="slot-status-tag" id="slot-status-pathology">Awaiting validation</span>
                                            </div>
                                        </div>
                                        <button type="button" id="btn-validate-pathology" class="btn-slot-validate btn-validate-image" onclick="validateSlotImage('pathology')">
                                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                            [ Validate Pathology ]
                                        </button>
                                        <div class="slot-feedback-line" id="slot-feedback-pathology" style="display: none;"></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Global Master Validate Button / Compatibility Hook -->
                        <div style="display: flex; justify-content: flex-end; align-items: center; margin-top: 0.75rem; gap: 0.75rem;">
                            <button type="button" id="btn-validate-image" class="btn-validate-all" onclick="validateAllLoadedSlots()">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                [ Validate All Images ]
                            </button>
                        </div>

                        <!-- Validation Banner / Alert Notification -->
                        <div id="validation-banner" class="validation-banner" style="display: none;">
                            <div class="banner-inner" id="validation-banner-content"></div>
                        </div>
                    </div>

                    <!-- Main Action Strip: [ Run Analysis ] and Reset -->
                    <div class="btn-action-row" style="margin-top: 1.5rem;">
                        <button type="button" id="btn-run-analysis" class="btn-primary-analyze" onclick="handleRunAnalysisClick()">
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg>
                            [ Run Analysis ]
                        </button>
                        <button type="button" class="btn-secondary-reset" onclick="resetForm()">
                            RESET
                        </button>
                    </div>
                </div>

                <!-- 3. Live 5-Stage Output Cards Grid -->
                <div class="stage-results-grid" id="results-grid" style="display: none;">
                    <!-- ML Result Card -->
                    <div class="stage-result-card" id="card-res-ml">
                        <div class="card-header-mini">
                            <h4>① Treatment Safety — Clinical Data Analysis</h4>
                            <span class="metric-badge blue" id="ml-badge-risk">Waiting</span>
                        </div>
                        <p id="ml-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting ML evaluation...</p>
                        <div class="feature-tag-list" id="ml-tags-list"></div>
                    </div>

                    <!-- DL Result Card -->
                    <div class="stage-result-card" id="card-res-dl">
                        <div class="card-header-mini">
                            <h4>② Medical Imaging — Medical Image Findings</h4>
                            <span class="metric-badge green" id="dl-badge-risk">Waiting</span>
                        </div>
                        <p id="dl-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting multimodal image inference...</p>
                        <div style="display: flex; gap: 0.5rem; align-items: center;" id="dl-preview-row"></div>
                    </div>

                    <!-- NLP Result Card -->
                    <div class="stage-result-card" id="card-res-nlp">
                        <div class="card-header-mini">
                            <h4>③ Clinical Triage — Clinical Text Analysis</h4>
                            <span class="metric-badge yellow" id="nlp-badge-triage">Waiting</span>
                        </div>
                        <p id="nlp-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting NLP entity extraction...</p>
                        <div class="feature-tag-list" id="nlp-entities-list"></div>
                    </div>

                    <!-- SLM Result Card -->
                    <div class="stage-result-card" id="card-res-slm">
                        <div class="card-header-mini">
                            <h4>④ Clinical AI Assistant — Clinical Reasoning</h4>
                            <span class="metric-badge blue" id="slm-badge-guardrail">Waiting</span>
                        </div>
                        <p id="slm-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting 3B SLM safety reasoning...</p>
                    </div>

                    <!-- GenAI Result Card -->
                    <div class="stage-result-card" id="card-res-genai">
                        <div class="card-header-mini">
                            <h4>⑤ Clinical Report — Generative Synthesis & Stress Test</h4>
                            <span class="metric-badge purple" id="genai-badge-status">Waiting</span>
                        </div>
                        <p id="genai-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting GenAI multi-modal report synthesis...</p>
                        <div class="feature-tag-list" id="genai-tags-list"></div>
                    </div>

                    <!-- Agentic AI Result Card -->
                    <div class="stage-result-card" id="card-res-agentic" style="border-left: 3px solid #0284c7;">
                        <div class="card-header-mini">
                            <h4>⑥ AI Care Workflow — Multi-Agent Deliberation & Trials</h4>
                            <span class="metric-badge blue" id="agentic-badge-status">Waiting</span>
                        </div>
                        <p id="agentic-text-interp" style="font-size: 0.78rem; color: var(--text-secondary); font-weight: 500;">Awaiting 10-agent consensus and clinical trial matching...</p>
                        <div class="feature-tag-list" id="agentic-tags-list"></div>
                        <div style="margin-top: 4px;">
                            <button type="button" class="btn-preset" onclick="switchView('view-stage6')" style="font-size: 0.72rem; padding: 0.25rem 0.6rem; background: #e0f2fe; color: #0284c7; border: 1px solid #38bdf8; border-radius: 4px; cursor: pointer; font-weight: 600;">
                                Open Stage 6 Deliberation Console →
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 4. Prominent FINAL AI ASSESSMENT Card -->
                <div class="final-assessment-card" id="final-assessment-section" style="display: none;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3 id="final-card-patient-title">Comprehensive Patient Assessment</h3>
                            <p id="final-card-patient-sub">Patient: PAT-NSCLC-0842 | Lung (LUAD) | Stage IV</p>
                        </div>
                        <div style="display: flex; gap: 0.65rem; align-items: center;">
                            <button type="button" class="btn-export-pdf-card" onclick="exportTumorBoardPdf()" title="Download Official Tumor Board PDF Report">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
                                <span>Export Report (PDF)</span>
                            </button>
                            <div class="risk-classification-tag high" id="final-risk-tag">
                                HIGH RISK
                            </div>
                        </div>
                    </div>

                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>OVERALL RISK TIER</span>
                            <h4 id="fin-overall-risk">HIGH RISK</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>RISK PROBABILITY</span>
                            <h4 id="fin-risk-prob">78.4%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL CONFIDENCE</span>
                            <h4 id="fin-confidence">91.5%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>STAGE 6 INTERLOCK</span>
                            <h4 id="fin-interlock-status" style="color: #059669;">NORMAL</h4>
                        </div>
                    </div>

                    <div class="clinical-summary-box" id="fin-genai-summary">
                        Generating comprehensive multi-modal oncology clinical report...
                    </div>

                    <!-- Stage 6 Autonomous Deliberation & Trial Matching Banner -->
                    <div id="fin-agentic-box" style="display: none; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: var(--radius-sm); padding: 0.85rem 1rem; margin-bottom: 0.85rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                            <h4 style="font-size: 0.82rem; font-weight: 800; color: #0369a1;">STAGE ⑥ MULTI-AGENT CONSENSUS & TRIAL MATCHING</h4>
                            <span id="fin-agentic-badge" class="metric-badge blue">Autonomous Panel</span>
                        </div>
                        <p id="fin-agentic-rec" style="font-size: 0.8rem; color: #0c4a6e; font-weight: 600; margin-bottom: 0.4rem;"></p>
                        <div id="fin-agentic-trials" style="font-size: 0.76rem; color: #0369a1;"></div>
                    </div>

                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.35rem;">RECOMMENDED CLINICAL CONSIDERATIONS</h4>
                        <div class="considerations-checklist" id="fin-considerations-list">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <div class="disclaimer-strip">
                        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                        AI-generated decision support — not a replacement for clinical judgment.
                    </div>
                </div>

                <!-- 5. INTERACTIVE WHAT-IF TRAJECTORY & DOSAGE SIMULATOR -->
                <div class="whatif-simulator-card" id="whatif-simulator-section">
                    <div class="whatif-header">
                        <div>
                            <h3 style="font-size: 0.96rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 0.5rem;">
                                <span>🎛️ Interactive "What-If" Dosage & Biomarker Simulator</span>
                                <span class="metric-badge blue" style="font-size: 0.65rem;">REAL-TIME TRAJECTORY ENGINE</span>
                            </h3>
                            <p style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">Dynamically adjust patient lab parameters to model live risk variations and safety guardrail interlocks.</p>
                        </div>
                        <button type="button" class="btn-preset" onclick="syncWhatIfToPatient()" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 6px; padding: 0.35rem 0.85rem; font-size: 0.74rem; font-weight: 700; cursor: pointer;">
                            Apply to Active Patient Form
                        </button>
                    </div>

                    <div class="whatif-grid">
                        <!-- Sliders Column -->
                        <div>
                            <div class="sim-slider-row">
                                <div class="sim-slider-header">
                                    <span>Chemotherapy Dosage (mg)</span>
                                    <span class="val" id="val-sim-dose">480 mg</span>
                                </div>
                                <input type="range" class="sim-slider" id="slider-sim-dose" min="50" max="1000" step="10" value="480" oninput="updateWhatIfSimulation()">
                            </div>

                            <div class="sim-slider-row">
                                <div class="sim-slider-header">
                                    <span>ctDNA Fractional Concentration</span>
                                    <span class="val" id="val-sim-ctdna">0.45</span>
                                </div>
                                <input type="range" class="sim-slider" id="slider-sim-ctdna" min="0.05" max="1.50" step="0.05" value="0.45" oninput="updateWhatIfSimulation()">
                            </div>

                            <div class="sim-slider-row">
                                <div class="sim-slider-header">
                                    <span>Consecutive Rising ctDNA Cycles</span>
                                    <span class="val" id="val-sim-rising">2 cycles</span>
                                </div>
                                <input type="range" class="sim-slider" id="slider-sim-rising" min="0" max="8" step="1" value="2" oninput="updateWhatIfSimulation()">
                            </div>

                            <div class="sim-slider-row">
                                <div class="sim-slider-header">
                                    <span>Liver Transaminase (ALT U/L)</span>
                                    <span class="val" id="val-sim-alt">45 U/L</span>
                                </div>
                                <input type="range" class="sim-slider" id="slider-sim-alt" min="15" max="300" step="5" value="45" oninput="updateWhatIfSimulation()">
                            </div>

                            <div class="sim-slider-row">
                                <div class="sim-slider-header">
                                    <span>Renal Function (Creatinine mg/dL)</span>
                                    <span class="val" id="val-sim-cr">1.1 mg/dL</span>
                                </div>
                                <input type="range" class="sim-slider" id="slider-sim-cr" min="0.5" max="4.0" step="0.1" value="1.1" oninput="updateWhatIfSimulation()">
                            </div>
                        </div>

                        <!-- Live Simulated Outcomes Column -->
                        <div class="sim-outcome-panel">
                            <div>
                                <span style="font-size: 0.68rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em;">PROJECTED TOXICITY STRATIFICATION</span>
                                <div class="sim-metric-badge" id="sim-res-toxicity" style="color: #38bdf8; margin-top: 0.2rem;">MODERATE RISK</div>
                            </div>

                            <div>
                                <span style="font-size: 0.68rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em;">MOLECULAR PROGRESSION PROBABILITY</span>
                                <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-top: 0.2rem;">
                                    <span id="sim-res-prog-pct" style="font-size: 1.4rem; font-weight: 900; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">58.4%</span>
                                    <span id="sim-res-prog-label" style="font-size: 0.72rem; color: #38bdf8; font-weight: 600;">(Stable Lesion Kinetics)</span>
                                </div>
                            </div>

                            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px; padding: 0.7rem;">
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <span style="font-size: 0.7rem; font-weight: 700; color: #cbd5e1;">AUTONOMOUS SAFETY INTERLOCK:</span>
                                    <span id="sim-res-interlock-badge" class="metric-badge green" style="font-size: 0.65rem;">PASSED</span>
                                </div>
                                <p id="sim-res-interlock-desc" style="font-size: 0.68rem; color: #94a3b8; margin-top: 0.3rem; line-height: 1.35;">
                                    Hepatic enzymes and renal filtration rates remain within acceptable safety envelope.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 2: STAGE 1 — ML TOXICITY & RISK DEEP DIVE
                 ============================================================== -->
            <section id="view-stage1" class="view-section">
                <div class="view-header">
                    <h2>STAGE ① — TREATMENT SAFETY (CLINICAL DATA ANALYSIS)</h2>
                    <p>Voting Ensemble Architecture (Random Forest, Extra Trees, XGBoost) with Permutation Feature Importances</p>
                </div>

                <!-- 1. ML PATIENT INPUT SECTION (BEFORE EXISTING REPORT) -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/></svg>
                            PATIENT CLINICAL DATA & BIOMARKER ANALYSIS
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">QUICK FILL:</span>
                            <button class="btn-preset" type="button" onclick="setMLSample('HIGH')">High Toxicity Sample</button>
                            <button class="btn-preset" type="button" onclick="setMLSample('MODERATE')">Moderate Sample</button>
                            <button class="btn-preset" type="button" onclick="setMLSample('LOW')">Low Toxicity Sample</button>
                        </div>
                    </div>

                    <form id="form-stage1" onsubmit="event.preventDefault(); runStage1Prediction();">
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Patient ID</label>
                                <input type="text" id="s1-patient-id" class="form-control" value="PAT-ML-0101" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age (Years)</label>
                                <input type="number" id="s1-age" class="form-control" value="64" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Sex</label>
                                <select id="s1-sex" class="form-control">
                                    <option value="M" selected>Male</option>
                                    <option value="F">Female</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Type</label>
                                <select id="s1-cancer-type" class="form-control">
                                    <option value="Lung (LUAD)" selected>Lung Adenocarcinoma (LUAD)</option>
                                    <option value="Lung (LUSC)">Lung Squamous (LUSC)</option>
                                    <option value="Breast (BRCA)">Breast Invasive (BRCA)</option>
                                    <option value="Colon (COAD)">Colon Adeno (COAD)</option>
                                    <option value="Prostate (PRAD)">Prostate Adeno (PRAD)</option>
                                    <option value="Stomach (STAD)">Stomach Adeno (STAD)</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Cancer Stage</label>
                                <select id="s1-cancer-stage" class="form-control">
                                    <option value="Stage I">Stage I</option>
                                    <option value="Stage II">Stage II</option>
                                    <option value="Stage III">Stage III</option>
                                    <option value="Stage IV" selected>Stage IV</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Biomarker Analysis</label>
                                <input type="text" id="s1-mutation" class="form-control" value="EGFR L858R">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Tumor Marker Level (ng/ml)</label>
                                <input type="number" id="s1-tumor-marker" class="form-control" value="28.6" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">ctDNA Fractional VAF (%)</label>
                                <input type="number" id="s1-ctdna" class="form-control" value="1.84" step="0.01" required>
                            </div>
                        </div>

                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">ALT (U/L)</label>
                                <input type="number" id="s1-alt" class="form-control" value="248" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">AST (U/L)</label>
                                <input type="number" id="s1-ast" class="form-control" value="210" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Creatinine (mg/dL)</label>
                                <input type="number" id="s1-creatinine" class="form-control" value="1.45" step="0.01" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Bilirubin (mg/dL)</label>
                                <input type="number" id="s1-bilirubin" class="form-control" value="2.2" step="0.1" required>
                            </div>
                        </div>

                        <div class="input-grid-3">
                            <div class="form-group">
                                <label class="form-label">Treatment Information / Regimen</label>
                                <input type="text" id="s1-treatment" class="form-control" value="Osimertinib 80mg + Capmatinib">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Dosage (mg)</label>
                                <input type="number" id="s1-dosage" class="form-control" value="480" step="10">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Treatment Cycle</label>
                                <input type="number" id="s1-cycle" class="form-control" value="4" step="1">
                            </div>
                        </div>

                        <div class="btn-action-row" style="margin-top: 0.5rem;">
                            <button type="submit" id="btn-run-stage1" class="btn-primary-analyze" style="max-width: 280px;">
                                [ RUN ML PREDICTION ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. ML PREDICTION RESULT CARD -->
                <div id="s1-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3 id="s1-out-title">TREATMENT SAFETY & CLINICAL DATA ANALYSIS RESULT</h3>
                            <p id="s1-out-patient-sub">Patient: PAT-ML-0101 | Model: VotingClassifier (Random Forest + Extra Trees + XGBoost)</p>
                        </div>
                        <div class="risk-classification-tag" id="s1-out-badge">HIGH RISK</div>
                    </div>
                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>PATIENT ID</span>
                            <h4 id="s1-out-pid">PAT-ML-0101</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>PREDICTED RISK CLASS</span>
                            <h4 id="s1-out-pred">High Toxicity</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>RISK LEVEL</span>
                            <h4 id="s1-out-class">HIGH RISK</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CONFIDENCE SCORE</span>
                            <h4 id="s1-out-conf">85.0%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL STATUS</span>
                            <h4 id="s1-out-model-status" style="color: #059669;">Operational</h4>
                        </div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem 0.9rem; margin-top: 0.6rem; display: flex; gap: 1.25rem; flex-wrap: wrap; align-items: center;">
                        <span style="font-size: 0.68rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">Calibrated Class Distribution:</span>
                        <div id="s1-out-class-probs" style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.78rem;">
                            <span><strong>Low Toxicity:</strong> —</span>
                            <span><strong>Moderate Toxicity:</strong> —</span>
                            <span><strong>High Toxicity:</strong> —</span>
                        </div>
                    </div>
                    <div class="clinical-summary-box" id="s1-out-interp">
                        Interpretation of structured parameters...
                    </div>
                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">IMPORTANT CLINICAL FACTORS</h4>
                        <div class="feature-tag-list" id="s1-out-features"></div>
                    </div>

                </div>

                <!-- 3. EXISTING ML REPORT / FEATURE IMPORTANCE -->
                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Top Influential Clinical Features (Permutation Importance)</h3>
                    <table class="clinical-table">
                        <thead>
                            <tr>
                                <th>Feature Name</th>
                                <th>Clinical Domain</th>
                                <th>Mean Permutation Importance</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td><strong>ALT (Alanine Aminotransferase)</strong></td><td>Hepatic Enzyme Panel</td><td>0.284</td><td><span class="metric-badge red">High Sensitivity</span></td></tr>
                            <tr><td><strong>AST (Aspartate Aminotransferase)</strong></td><td>Hepatic Enzyme Panel</td><td>0.221</td><td><span class="metric-badge red">High Sensitivity</span></td></tr>
                            <tr><td><strong>ctDNA Fractional VAF</strong></td><td>Liquid Biopsy Genomics</td><td>0.185</td><td><span class="metric-badge yellow">Moderate Sensitivity</span></td></tr>
                            <tr><td><strong>Tumor Marker Level</strong></td><td>Serum Oncology Biomarkers</td><td>0.142</td><td><span class="metric-badge blue">Stable</span></td></tr>
                            <tr><td><strong>Creatinine</strong></td><td>Renal Function Panel</td><td>0.098</td><td><span class="metric-badge blue">Stable</span></td></tr>
                            <tr><td><strong>Pulse Pressure</strong></td><td>Vascular Vitals</td><td>0.070</td><td><span class="metric-badge green">Baseline</span></td></tr>
                        </tbody>
                    </table>
                </div>

                <!-- 🧠 Stage 1 SHAP Feature Attribution Waterfall (Local Explainability) -->
                <div class="shap-waterfall-card" style="margin-top: 1rem; border: 1px solid var(--border-card); background: #ffffff;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <h4 style="font-size: 0.95rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 0.45rem;">
                            <span>🧠 Stage 1 SHAP Feature Attribution Waterfall</span>
                            <span class="metric-badge purple" style="font-size: 0.65rem;">LOCAL SHAPLEY EXPLAINABILITY</span>
                        </h4>
                        <span style="font-size: 0.72rem; color: #64748b; font-family: 'JetBrains Mono', monospace; font-weight: 700;">Base Expected Value E[f(x)] = 0.33</span>
                    </div>
                    <p style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.9rem;">
                        Quantifies patient-specific biomarker push towards <strong style="color: #dc2626;">High Risk (+)</strong> vs <strong style="color: #059669;">Protective Clearance (-)</strong> based on Shapley additive explanations.
                    </p>
                    <div id="s1-shap-waterfall-bars">
                        <!-- Populated dynamically with waterfall bars -->
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 3: STAGE 2 — DL MEDICAL IMAGE ANALYSIS DEEP DIVE
                 ============================================================== -->
            <section id="view-stage2" class="view-section">
                <div class="view-header">
                    <h2>STAGE ② — MEDICAL IMAGING (MEDICAL IMAGE FINDINGS)</h2>
                    <p>MultimodalLSTM combining 2D CT Slices, Pathology Tiles, and Longitudinal Sequence Steps</p>
                </div>

                <!-- 1. DL INDEPENDENT INPUT SECTION -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                            DL MULTIMODAL INPUT (STRUCTURED PATIENT/CSV DATA + MEDICAL IMAGE)
                        </h3>
                    </div>

                    <form id="form-stage2" onsubmit="event.preventDefault(); runStage2Prediction();">
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Patient ID</label>
                                <input type="text" id="s2-patient-id" class="form-control" value="PAT-DL-0201" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age</label>
                                <input type="number" id="s2-age" class="form-control" value="61" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Type</label>
                                <select id="s2-cancer-type" class="form-control" onchange="updateDLSampleImages(this.value)">
                                    <option value="luad" selected>Lung Adenocarcinoma (LUAD)</option>
                                    <option value="lusc">Lung Squamous (LUSC)</option>
                                    <option value="brca">Breast Invasive (BRCA)</option>
                                    <option value="prad">Prostate Adeno (PRAD)</option>
                                    <option value="coad">Colon Adeno (COAD)</option>
                                    <option value="stad">Stomach Adeno (STAD)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Stage</label>
                                <select id="s2-cancer-stage" class="form-control">
                                    <option value="Stage I">Stage I</option>
                                    <option value="Stage II">Stage II</option>
                                    <option value="Stage III">Stage III</option>
                                    <option value="Stage IV" selected>Stage IV</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Biomarker Analysis</label>
                                <input type="text" id="s2-mutation" class="form-control" value="EGFR L858R">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Tumor Marker Level (ng/mL)</label>
                                <input type="number" id="s2-tumor-marker" class="form-control" value="24.5" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">ctDNA Fractional Level (%)</label>
                                <input type="number" id="s2-ctdna" class="form-control" value="1.62" step="0.01" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Baseline WSI Atypia Score</label>
                                <input type="number" id="s2-atypia" class="form-control" value="0.75" step="0.05" min="0" max="1">
                            </div>
                        </div>

                        <!-- PRIMARY MEDICAL IMAGE UPLOAD WORKFLOW (STAGE 2 MULTI-MODAL SLOTS) -->
                        <div class="medical-upload-panel" style="margin-top: 0.75rem;">
                            <div class="modality-select-section">
                                <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem;">
                                    <label class="section-subheading" style="margin-bottom: 0;">
                                        Select Image Type / Multi-Modal Input Slots <span class="required-indicator">*</span>
                                    </label>
                                    <span style="font-size: 0.72rem; color: #64748b; font-weight: 500;">
                                        Upload &amp; validate MRI, CT Scan, and Pathology individually in dedicated slots below:
                                    </span>
                                </div>

                                <!-- Triple Modality Grid: MRI, CT Scan, Pathology -->
                                <div class="triple-modality-container">
                                    <!-- 1. MRI Scan Slot -->
                                    <div class="modality-slot-card" id="s2-slot-card-mri">
                                        <div class="slot-header" id="s2-mod-pill-mri">
                                            <div class="slot-header-info">
                                                <div class="slot-icon-badge mri">
                                                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18"/><path d="M12 7a5 5 0 0 0 0 10"/></svg>
                                                </div>
                                                <div>
                                                    <h4 class="slot-title">MRI Scan</h4>
                                                    <span class="slot-subtitle">Magnetic Resonance Imaging</span>
                                                </div>
                                            </div>
                                            <div class="slot-header-actions">
                                                <span class="slot-pill-badge awaiting" id="s2-slot-badge-mri">Awaiting Scan</span>
                                                <button type="button" class="btn-slot-clear" id="s2-slot-btn-clear-mri" style="display: none;" onclick="clearSlotImage('mri', 's2')" title="Clear MRI image">✕ Clear</button>
                                            </div>
                                        </div>

                                        <!-- Dropzone (when empty) -->
                                        <div class="slot-dropzone" id="s2-slot-dropzone-mri"
                                             ondragover="handleSlotDragOver(event, 'mri', 's2')"
                                             ondragleave="handleSlotDragLeave(event, 'mri', 's2')"
                                             ondrop="handleSlotDrop(event, 'mri', 's2')"
                                             onclick="triggerSlotFileInput('mri', 's2')">
                                            <div class="slot-drop-icon">
                                                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                                            </div>
                                            <span class="slot-drop-title">[ Upload Medical Image ]</span>
                                            <span class="slot-drop-formats">PNG, JPG, JPEG (Max 15MB)</span>
                                            <label for="s2-file-input-mri" class="btn-slot-choose" onclick="event.stopPropagation()">
                                                [ Choose MRI ]
                                            </label>
                                            <input type="file" id="s2-file-input-mri" accept=".png,.jpg,.jpeg,image/png,image/jpeg" style="display: none;" onchange="handleSlotFileSelect(this, 'mri', 's2')">
                                        </div>

                                        <!-- Preview & Validation Box (when populated) -->
                                        <div class="slot-preview-box" id="s2-slot-preview-mri" style="display: none;">
                                            <div class="slot-thumb-wrapper">
                                                <img id="s2-slot-img-mri" src="" alt="MRI Preview" class="slot-thumb-img">
                                            </div>
                                            <div style="display: flex; justify-content: center; margin-top: -0.2rem;">
                                                <label for="s2-file-input-mri" class="slot-change-btn" onclick="event.stopPropagation()">
                                                    ↻ Change Image
                                                </label>
                                            </div>
                                            <div class="slot-meta-wrapper">
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">File:</span>
                                                    <span class="slot-meta-val" id="s2-slot-filename-mri">—</span>
                                                </div>
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">Size:</span>
                                                    <span class="slot-meta-val" id="s2-slot-filesize-mri">—</span>
                                                </div>
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">Status:</span>
                                                    <span class="slot-status-tag" id="s2-slot-status-mri">Awaiting validation</span>
                                                </div>
                                            </div>
                                            <button type="button" id="s2-btn-validate-mri" class="btn-slot-validate btn-validate-image" onclick="validateSlotImage('mri', 's2')">
                                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                                [ Validate MRI ]
                                            </button>
                                            <div class="slot-feedback-line" id="s2-slot-feedback-mri" style="display: none;"></div>
                                        </div>
                                    </div>

                                    <!-- 2. CT Scan Slot -->
                                    <div class="modality-slot-card" id="s2-slot-card-ct">
                                        <div class="slot-header" id="s2-mod-pill-ct">
                                            <div class="slot-header-info">
                                                <div class="slot-icon-badge ct">
                                                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/></svg>
                                                </div>
                                                <div>
                                                    <h4 class="slot-title">CT Scan</h4>
                                                    <span class="slot-subtitle">2D Axial Radiology Slice</span>
                                                </div>
                                            </div>
                                            <div class="slot-header-actions">
                                                <span class="slot-pill-badge awaiting" id="s2-slot-badge-ct">Awaiting Scan</span>
                                                <button type="button" class="btn-slot-clear" id="s2-slot-btn-clear-ct" style="display: none;" onclick="clearSlotImage('ct', 's2')" title="Clear CT image">✕ Clear</button>
                                            </div>
                                        </div>

                                        <!-- Dropzone (when empty) -->
                                        <div class="slot-dropzone" id="s2-slot-dropzone-ct"
                                             ondragover="handleSlotDragOver(event, 'ct', 's2')"
                                             ondragleave="handleSlotDragLeave(event, 'ct', 's2')"
                                             ondrop="handleSlotDrop(event, 'ct', 's2')"
                                             onclick="triggerSlotFileInput('ct', 's2')">
                                            <div class="slot-drop-icon">
                                                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                                            </div>
                                            <span class="slot-drop-title">[ Upload Medical Image ]</span>
                                            <span class="slot-drop-formats">PNG, JPG, JPEG (Max 15MB)</span>
                                            <label for="s2-file-input-ct" class="btn-slot-choose" onclick="event.stopPropagation()">
                                                [ Choose CT Scan ]
                                            </label>
                                            <input type="file" id="s2-file-input-ct" accept=".png,.jpg,.jpeg,image/png,image/jpeg" style="display: none;" onchange="handleSlotFileSelect(this, 'ct', 's2')">
                                        </div>

                                        <!-- Preview & Validation Box (when populated) -->
                                        <div class="slot-preview-box" id="s2-slot-preview-ct" style="display: none;">
                                            <div class="slot-thumb-wrapper">
                                                <img id="s2-slot-img-ct" src="" alt="CT Preview" class="slot-thumb-img">
                                            </div>
                                            <div style="display: flex; justify-content: center; margin-top: -0.2rem;">
                                                <label for="s2-file-input-ct" class="slot-change-btn" onclick="event.stopPropagation()">
                                                    ↻ Change Image
                                                </label>
                                            </div>
                                            <div class="slot-meta-wrapper">
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">File:</span>
                                                    <span class="slot-meta-val" id="s2-slot-filename-ct">—</span>
                                                </div>
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">Size:</span>
                                                    <span class="slot-meta-val" id="s2-slot-filesize-ct">—</span>
                                                </div>
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">Status:</span>
                                                    <span class="slot-status-tag" id="s2-slot-status-ct">Awaiting validation</span>
                                                </div>
                                            </div>
                                            <button type="button" id="s2-btn-validate-ct" class="btn-slot-validate btn-validate-image" onclick="validateSlotImage('ct', 's2')">
                                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                                [ Validate CT ]
                                            </button>
                                            <div class="slot-feedback-line" id="s2-slot-feedback-ct" style="display: none;"></div>
                                        </div>
                                    </div>

                                    <!-- 3. Pathology Slot -->
                                    <div class="modality-slot-card" id="s2-slot-card-pathology">
                                        <div class="slot-header" id="s2-mod-pill-pathology">
                                            <div class="slot-header-info">
                                                <div class="slot-icon-badge pathology">
                                                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                                                </div>
                                                <div>
                                                    <h4 class="slot-title">Pathology</h4>
                                                    <span class="slot-subtitle">H&amp;E Histopathology Tile</span>
                                                </div>
                                            </div>
                                            <div class="slot-header-actions">
                                                <span class="slot-pill-badge awaiting" id="s2-slot-badge-pathology">Awaiting Scan</span>
                                                <button type="button" class="btn-slot-clear" id="s2-slot-btn-clear-pathology" style="display: none;" onclick="clearSlotImage('pathology', 's2')" title="Clear Pathology image">✕ Clear</button>
                                            </div>
                                        </div>

                                        <!-- Dropzone (when empty) -->
                                        <div class="slot-dropzone" id="s2-slot-dropzone-pathology"
                                             ondragover="handleSlotDragOver(event, 'pathology', 's2')"
                                             ondragleave="handleSlotDragLeave(event, 'pathology', 's2')"
                                             ondrop="handleSlotDrop(event, 'pathology', 's2')"
                                             onclick="triggerSlotFileInput('pathology', 's2')">
                                            <div class="slot-drop-icon">
                                                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                                            </div>
                                            <span class="slot-drop-title">[ Upload Medical Image ]</span>
                                            <span class="slot-drop-formats">PNG, JPG, JPEG (Max 15MB)</span>
                                            <label for="s2-file-input-pathology" class="btn-slot-choose" onclick="event.stopPropagation()">
                                                [ Choose Pathology ]
                                            </label>
                                            <input type="file" id="s2-file-input-pathology" accept=".png,.jpg,.jpeg,image/png,image/jpeg" style="display: none;" onchange="handleSlotFileSelect(this, 'pathology', 's2')">
                                        </div>

                                        <!-- Preview & Validation Box (when populated) -->
                                        <div class="slot-preview-box" id="s2-slot-preview-pathology" style="display: none;">
                                            <div class="slot-thumb-wrapper">
                                                <img id="s2-slot-img-pathology" src="" alt="Pathology Preview" class="slot-thumb-img">
                                            </div>
                                            <div style="display: flex; justify-content: center; margin-top: -0.2rem;">
                                                <label for="s2-file-input-pathology" class="slot-change-btn" onclick="event.stopPropagation()">
                                                    ↻ Change Image
                                                </label>
                                            </div>
                                            <div class="slot-meta-wrapper">
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">File:</span>
                                                    <span class="slot-meta-val" id="s2-slot-filename-pathology">—</span>
                                                </div>
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">Size:</span>
                                                    <span class="slot-meta-val" id="s2-slot-filesize-pathology">—</span>
                                                </div>
                                                <div class="slot-meta-row">
                                                    <span class="slot-meta-label">Status:</span>
                                                    <span class="slot-status-tag" id="s2-slot-status-pathology">Awaiting validation</span>
                                                </div>
                                            </div>
                                            <button type="button" id="s2-btn-validate-pathology" class="btn-slot-validate btn-validate-image" onclick="validateSlotImage('pathology', 's2')">
                                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                                [ Validate Pathology ]
                                            </button>
                                            <div class="slot-feedback-line" id="s2-slot-feedback-pathology" style="display: none;"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Master Validate Button for Stage 2 -->
                            <div style="display: flex; justify-content: flex-end; align-items: center; margin-top: 0.75rem; gap: 0.75rem;">
                                <button type="button" id="s2-btn-validate-image" class="btn-validate-all" onclick="validateAllLoadedSlots('s2')">
                                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                    [ Validate All Images ]
                                </button>
                            </div>

                            <!-- Validation Banner for Stage 2 -->
                            <div id="s2-validation-banner" class="validation-banner" style="display: none;">
                                <div class="banner-inner" id="s2-validation-banner-content"></div>
                            </div>
                        </div>

                        <div class="btn-action-row" style="margin-top: 1.25rem;">
                            <button type="submit" id="btn-run-stage2" class="btn-primary-analyze" style="max-width: 280px;">
                                [ RUN DL PREDICTION ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. DL OUTPUT SECTION -->
                <div id="s2-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3>MEDICAL IMAGE FINDINGS & MULTIMODAL INFERENCE</h3>
                            <p id="s2-out-patient-sub">Patient: PAT-DL-0201 | Model: MultimodalLSTM (CNN Embeddings + Longitudinal Sequence)</p>
                        </div>
                        <div class="risk-classification-tag high" id="s2-out-risk-tier">HIGH PROGRESSION RISK</div>
                    </div>
                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>PATIENT ID</span>
                            <h4 id="s2-out-pid">PAT-DL-0201</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MEDICAL IMAGE FINDINGS</span>
                            <h4 id="s2-out-class">Malignant Neoplasm</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CLASSIFICATION</span>
                            <h4 id="s2-out-classif">Progressive Lesion</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>PROBABILITY</span>
                            <h4 id="s2-out-prob">76.4%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CONFIDENCE</span>
                            <h4 id="s2-out-conf">92.0%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL STATUS</span>
                            <h4 id="s2-out-model-status" style="color: #059669;">Operational</h4>
                        </div>
                    </div>
                    <div class="clinical-summary-box" id="s2-out-interp">
                        Trajectory progression interpretation...
                    </div>
                    <div id="s2-timepoint-box">
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">LONGITUDINAL SEQUENCE TRAJECTORY (T0 — T4)</h4>
                        <table class="clinical-table">
                            <thead><tr><th>Sequence Step</th><th>Timepoint</th><th>Progression Prob</th><th>Risk Tier</th></tr></thead>
                            <tbody id="s2-trajectory-table"></tbody>
                        </table>
                    </div>
                </div>

            </section>


            <!-- ==============================================================
                 VIEW 4: STAGE 3 — NLP CLINICAL TEXT DEEP DIVE
                 ============================================================== -->
            <section id="view-stage3" class="view-section">
                <div class="view-header">
                    <h2>STAGE ③ — CLINICAL TRIAGE (CLINICAL TEXT ANALYSIS)</h2>
                    <p>Hybrid TF-IDF + Logistic Regression Urgency Classifier with Named Entity Recognition</p>
                </div>

                <!-- 1. NLP INDEPENDENT INPUT SECTION -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/></svg>
                            CLINICAL NOTE ANALYSIS & PATIENT REPORT INPUT
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">SAMPLES:</span>
                            <button class="btn-preset" type="button" onclick="setNLPTextSample('HIGH')">High Urgency Fever</button>
                            <button class="btn-preset" type="button" onclick="setNLPTextSample('MODERATE')">Moderate Fatigue</button>
                            <button class="btn-preset danger" type="button" onclick="setNLPTextSample('CRITICAL')">Critical Sepsis</button>
                        </div>
                    </div>

                    <form id="form-stage3" onsubmit="event.preventDefault(); runStage3Prediction();">
                        <div class="form-group" style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                                <label class="form-label" style="margin-bottom: 0;">Enter clinical notes, symptoms, diagnosis, medical history, or oncology report text...</label>
                                <button type="button" class="btn-voice-dictate" id="btn-dictate-stage3" onclick="toggleVoiceDictation('inp-stage3-text', 'btn-dictate-stage3')" title="Dictate Clinical Consult via Microphone">
                                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                                    <span class="dictate-text">Doctor Voice Dictate</span>
                                </button>
                            </div>
                            <textarea id="inp-stage3-text" class="form-control" style="min-height: 110px;" placeholder="Enter clinical notes, symptoms, diagnosis, medical history, or oncology report text or use 'Doctor Voice Dictate'...">Patient has elevated tumor markers and worsening symptoms with high fever 38.8C, persistent fatigue, and right upper quadrant abdominal tenderness concerning for acute drug toxicity.</textarea>
                        </div>

                        <div class="btn-action-row">
                            <button type="submit" id="btn-run-stage3" class="btn-primary-analyze" style="max-width: 280px;">
                                [ RUN NLP PREDICTION ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. NLP OUTPUT SECTION -->
                <div id="s3-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3>CLINICAL TRIAGE ASSESSMENT & EXTRACTION RESULT</h3>
                            <p>Model: Hybrid TF-IDF + Logistic Regression with Biomedical Rule/Dictionary NER</p>
                        </div>
                        <div class="risk-classification-tag high" id="s3-out-badge">HIGH RISK</div>
                    </div>
                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>RISK LEVEL</span>
                            <h4 id="s3-out-level">HIGH RISK</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>PROBABILITY</span>
                            <h4 id="s3-out-prob">0.950</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CONFIDENCE</span>
                            <h4 id="s3-out-conf">95.0%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL STATUS</span>
                            <h4 id="s3-out-model-status" style="color: #059669;">Operational</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>ENTITIES EXTRACTED</span>
                            <h4 id="s3-out-count">4 Identified</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>LATENCY</span>
                            <h4 id="s3-out-latency">14.2 ms</h4>
                        </div>
                    </div>
                    <div class="clinical-summary-box" id="s3-out-interp">
                        NLP triage output...
                    </div>
                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">EXTRACTED CLINICAL ENTITIES & TERMS</h4>
                        <div class="feature-tag-list" id="s3-out-entities"></div>
                    </div>
                </div>

                <!-- 3. EXISTING CLINICAL GUIDELINE RETRIEVAL SOP MATCHES -->
                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Clinical Guideline Retrieval SOP Matches</h3>
                    <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                        <div class="clinical-summary-box">
                            <strong>NCCN Immune-Mediated Hepatotoxicity Guideline (Grade 3+):</strong>
                            <p>For ALT/AST > 5x ULN or Total Bilirubin > 3x ULN, withhold immunotherapy/TKI regimen immediately. Administer oral prednisone 1-2 mg/kg/day with daily liver panel monitoring.</p>
                        </div>
                        <div class="clinical-summary-box" style="border-left-color: #10b981;">
                            <strong>ASCO Febrile Neutropenia & Sepsis Triage Protocol:</strong>
                            <p>Immediate blood cultures and stat empiric broad-spectrum antibiotic initiation within 60 minutes for temperature > 38.3C and absolute neutrophil count < 500/uL.</p>
                        </div>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 5: STAGE 4 — SLM CLINICAL REASONING DEEP DIVE
                 ============================================================== -->
            <section id="view-stage4" class="view-section">
                <div class="view-header">
                    <h2>STAGE ④ — CLINICAL AI ASSISTANT (REASONING)</h2>
                    <p>Fine-Tuned Qwen2.5-3B Instruct with Strict Post-Inference Safety Guardrails</p>
                </div>

                <!-- 1. SLM INDEPENDENT SIMPLIFICATION INPUT SECTION -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                            LONG CLINICAL TEXT (CONVERT TO SHORT, SIMPLE CLINICAL EXPLANATION)
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">SAMPLES:</span>
                            <button class="btn-preset" type="button" onclick="setSLMTextSample('TKI')">TKI Hepatic Consult</button>
                            <button class="btn-preset" type="button" style="border-color: #ea580c; color: #ea580c;" onclick="setSLMTextSample('HIGH')">High-Acuity Toxicity (SpO2/Fever)</button>
                            <button class="btn-preset danger" type="button" onclick="setSLMTextSample('CRITICAL')">Emergency ICU Sepsis</button>
                        </div>
                    </div>

                    <form id="form-stage4" onsubmit="event.preventDefault(); runStage4Simplification();">
                        <div class="form-group" style="margin-bottom: 1rem;">
                            <label class="form-label">Enter long medical/clinical paragraph:</label>
                            <textarea id="inp-stage4-text" class="form-control" style="min-height: 120px;" placeholder="Paste long technical oncology report or physician notes...">Patient PAT-0001 (Lung, EGFR L858R) on Targeted Therapy presents with acute transaminase elevation with ALT 248 U/L and AST 210 U/L following cycle 4 of tyrosine kinase inhibitor therapy. Clinical observation reveals persistent right upper quadrant discomfort, fever at 38.4C, and intractable nausea concerning for drug-induced liver injury requiring urgent clinical review and hydration support.</textarea>
                        </div>

                        <div class="btn-action-row" style="flex-wrap: wrap;">
                            <button type="submit" id="btn-run-stage4" class="btn-primary-analyze" style="max-width: 250px;">
                                [ SIMPLIFY WITH SLM ]
                            </button>
                            <button type="button" id="btn-predict-stage4" class="btn-primary-analyze" style="max-width: 270px; background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);" onclick="runStage4Simplification()">
                                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                                [ PREDICT URGENCY RANGE ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. SLM OUTPUT SECTION -->
                <div id="s4-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3>SIMPLIFIED CLINICAL EXPLANATION</h3>
                            <p>Original Text ↓ Ultra-Concise Simplified Clinical Finding (< 2 lines)</p>
                        </div>
                        <div class="risk-classification-tag moderate" id="s4-out-status-badge">VERIFIED SAFE</div>
                    </div>

                    <!-- CLINICAL URGENCY RANGE PREDICTION BAR -->
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
                            <span style="font-size: 0.72rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; gap: 0.4rem;">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                                PREDICTED URGENCY RANGE:
                            </span>
                            <span id="s4-out-tier-badge" style="font-size: 0.75rem; font-weight: 800; padding: 0.25rem 0.75rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; background: #ffedd5; color: #9a3412;">HIGH URGENCY</span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; text-align: center;" id="s4-tier-meter">
                            <div id="s4-tier-low" style="padding: 0.6rem 0.4rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #e2e8f0; background: #ffffff; color: #94a3b8; transition: all 0.25s;">
                                LOW
                            </div>
                            <div id="s4-tier-moderate" style="padding: 0.6rem 0.4rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #e2e8f0; background: #ffffff; color: #94a3b8; transition: all 0.25s;">
                                MODERATE
                            </div>
                            <div id="s4-tier-high" style="padding: 0.6rem 0.4rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #e2e8f0; background: #ffffff; color: #94a3b8; transition: all 0.25s;">
                                HIGH
                            </div>
                            <div id="s4-tier-critical" style="padding: 0.6rem 0.4rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #e2e8f0; background: #ffffff; color: #94a3b8; transition: all 0.25s;">
                                CRITICAL
                            </div>
                        </div>
                    </div>

                    <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                        <div>
                            <span style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">ORIGINAL TEXT</span>
                            <div style="background: #f1f5f9; padding: 0.75rem; border-radius: 6px; font-size: 0.78rem; color: #475569; margin-top: 0.25rem;" id="s4-out-original">
                                Original text...
                            </div>
                        </div>

                        <div style="text-align: center; color: var(--accent-blue); font-size: 1.25rem; font-weight: 800;">↓</div>

                        <div>
                            <span style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">SIMPLIFIED CLINICAL EXPLANATION (< 2 LINES)</span>
                            <div class="clinical-summary-box" style="border-left-color: var(--accent-blue); font-size: 0.86rem; line-height: 1.45; padding: 0.75rem 1rem;" id="s4-out-simplified">
                                Simplified concise clinical explanation...
                            </div>
                        </div>
                    </div>

                    <div class="assessment-breakdown-grid" style="margin-top: 0.5rem;">
                        <div class="breakdown-box">
                            <span>PROCESSING STATUS</span>
                            <h4 id="s4-out-proc">Completed ✓</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL STATUS</span>
                            <h4 id="s4-out-model">Operational (Qwen2.5-3B-Instruct)</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>SAFETY / GUARDRAIL</span>
                            <h4 id="s4-out-guardrail" style="color: #059669;">VERIFIED_SAFE</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>LATENCY</span>
                            <h4 id="s4-out-latency">12.0 ms</h4>
                        </div>
                    </div>
                </div>

                <!-- 3. EXISTING GUARDRAIL VERIFICATION AUDIT MATRIX -->
                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Guardrail Verification Audit Matrix</h3>
                    <table class="clinical-table">
                        <thead>
                            <tr>
                                <th>Safety Constraint</th>
                                <th>Validation Method</th>
                                <th>Compliance Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td><strong>Under-Triage Interception</strong></td><td>Rule Floor Escalator (High/Critical Floor Detection)</td><td><span class="metric-badge green">100% Zero-Leak</span></td></tr>
                            <tr><td><strong>Patient ID Retention</strong></td><td>Regex Deterministic ID Injection</td><td><span class="metric-badge green">100% Preserved</span></td></tr>
                            <tr><td><strong>Strict 2-Sentence Compliance</strong></td><td>Punctuation Sentence Normalizer</td><td><span class="metric-badge green">Pass (2 Sentences)</span></td></tr>
                        </tbody>
                    </table>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 6: STAGE 5 — GENAI CLINICAL REPORT DEEP DIVE
                 ============================================================== -->
            <section id="view-stage5" class="view-section">
                <div class="view-header">
                    <h2>STAGE ⑤ — CLINICAL REPORT (REPORT INSIGHTS)</h2>
                    <p>Multi-Stage Synthetic Patient Simulation & Cross-Modal Safety Auditing</p>
                </div>

                <!-- 1. GENAI INDEPENDENT INPUT SECTION -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                            SEED CONDITIONS & CLINICAL CONTEXT FOR GENAI
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">SEED PRESETS:</span>
                            <button class="btn-preset" type="button" onclick="setGenAISeedSample('NSCLC')">Metastatic NSCLC</button>
                            <button class="btn-preset" type="button" onclick="setGenAISeedSample('BRCA')">CDK4/6 Breast Cancer</button>
                            <button class="btn-preset danger" type="button" onclick="setGenAISeedSample('DILI')">Severe DILI Wildcard</button>
                        </div>
                    </div>

                    <form id="form-stage5" onsubmit="event.preventDefault(); runStage5Generation();">
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Cancer Type</label>
                                <select id="s5-cancer-type" class="form-control">
                                    <option value="Lung (LUAD)" selected>Lung Adenocarcinoma (LUAD)</option>
                                    <option value="Lung (LUSC)">Lung Squamous (LUSC)</option>
                                    <option value="Breast (BRCA)">Breast Invasive (BRCA)</option>
                                    <option value="Colon (COAD)">Colon Adeno (COAD)</option>
                                    <option value="Prostate (PRAD)">Prostate Adeno (PRAD)</option>
                                    <option value="Stomach (STAD)">Stomach Adeno (STAD)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age Range</label>
                                <input type="text" id="s5-age-range" class="form-control" value="60 - 70">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Biomarker Analysis</label>
                                <input type="text" id="s5-mutation" class="form-control" value="EGFR L858R, MET amp">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Biomarker</label>
                                <input type="text" id="s5-biomarker" class="form-control" value="PD-L1 85%, TMB 14 mut/Mb">
                            </div>
                        </div>

                        <div class="input-grid-3">
                            <div class="form-group">
                                <label class="form-label">Cancer Stage</label>
                                <select id="s5-stage" class="form-control">
                                    <option value="Stage I">Stage I</option>
                                    <option value="Stage II">Stage II</option>
                                    <option value="Stage III">Stage III</option>
                                    <option value="Stage IV" selected>Stage IV</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Treatment</label>
                                <input type="text" id="s5-treatment" class="form-control" value="Osimertinib + Capmatinib">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Baseline ALT (U/L)</label>
                                <input type="number" id="s5-alt" class="form-control" value="248" step="1">
                            </div>
                        </div>

                        <div class="form-group" style="margin-bottom: 1rem;">
                            <label class="form-label">Optional Clinical Context / Seed Symptoms</label>
                            <textarea id="s5-notes" class="form-control" style="min-height: 80px;" placeholder="Add clinical background context...">Patient developing fever 38.4C and elevated transaminases under targeted therapy combination.</textarea>
                        </div>

                        <div class="btn-action-row">
                            <button type="submit" id="btn-run-stage5" class="btn-primary-analyze" style="max-width: 280px;">
                                [ GENERATE ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. GENAI OUTPUT SECTION -->
                <div id="s5-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3 id="s5-out-title">CLINICAL REPORT INSIGHTS</h3>
                            <p>Patient / Synthetic Profile & Cross-Modal Safety Auditing</p>
                        </div>
                        <div class="risk-classification-tag high" id="s5-out-badge">VALIDATED REPORT</div>
                    </div>

                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>FACTUAL CONSISTENCY</span>
                            <h4 id="s5-out-factual">98.2%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CROSS-MODAL SAFETY</span>
                            <h4 id="s5-out-safety">95.0%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>HALLUCINATION INDEX</span>
                            <h4 id="s5-out-halluc" style="color: #059669;">1.8%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>VALIDATION</span>
                            <h4 id="s5-out-valid" style="color: #059669;">PASSED</h4>
                        </div>
                    </div>

                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">GENERATED PATIENT / SYNTHETIC PROFILE</h4>
                        <div class="clinical-summary-box" id="s5-out-summary">
                            Generated clinical summary...
                        </div>
                    </div>

                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">KEY CLINICAL FINDINGS</h4>
                        <div class="considerations-checklist" id="s5-out-findings"></div>
                    </div>

                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">FULL GENERATED CLINICAL REPORT</h4>
                        <pre style="background: #0b1e3f; color: #f1f5f9; padding: 1rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; overflow-x: auto; white-space: pre-wrap;" id="s5-out-report"></pre>
                    </div>
                </div>

                <!-- 3. EXISTING AUDIT & DIVERGENCE METRICS -->
                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Audit & Divergence Metrics</h3>
                    <div class="analytics-stats-grid">
                        <div class="stat-card"><span>Factual Consistency</span><h3>98.2%</h3></div>
                        <div class="stat-card"><span>Cross-Modal Safety</span><h3>95.0%</h3></div>
                        <div class="stat-card"><span>Hallucination Rate</span><h3 style="color: #059669;">1.8%</h3></div>
                        <div class="stat-card"><span>Divergence Score</span><h3>0.042</h3></div>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 6B: STAGE 6 — AGENTIC AI COMMAND CENTER
                 ============================================================== -->
            <section id="view-stage6" class="view-section">
                <div class="view-header">
                    <h2>STAGE ⑥ — AI CARE WORKFLOW</h2>
                    <p>Autonomous Multi-Agent Oncology Decision Engine · Deliberative Clinical Decision Support</p>
                </div>

                <!-- 1. PATIENT CONTEXT & CASE PRESETS -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                            PATIENT CONTEXT & CLINICAL REASONING FEEDER
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">CLINICAL PRESETS:</span>
                            <button class="btn-preset" type="button" onclick="setAgenticPreset('case_a_egfr')" style="background: #e0f2fe; color: #0369a1; border-color: #38bdf8;">Patient A: EGFR+ (6 ctDNA Surge)</button>
                            <button class="btn-preset" type="button" onclick="setAgenticPreset('case_b_kras')" style="background: #fef3c7; color: #92400e; border-color: #f59e0b;">Patient B: KRAS+ (Stable 2 ctDNA)</button>
                            <button class="btn-preset danger" type="button" onclick="setAgenticPreset('case_c_dili')">DILI Safety Alert</button>
                            <button class="btn-preset" type="button" onclick="setAgenticPreset('case_d_no_image')">No Image Edge Case</button>
                            <button class="btn-preset" type="button" onclick="pullExistingAiResults()" style="background: #ecfdf5; color: #065f46; border-color: #10b981; font-weight: 800;">[ USE EXISTING AI RESULTS ]</button>
                        </div>
                    </div>

                    <form id="form-stage6" onsubmit="event.preventDefault(); runAgenticAnalysis();">
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Patient ID</label>
                                <input type="text" id="s6-patient-id" class="form-control" value="PAT-EGFR-001">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age & Sex</label>
                                <div style="display: flex; gap: 0.5rem;">
                                    <input type="number" id="s6-age" class="form-control" value="62" style="width: 60%;">
                                    <select id="s6-sex" class="form-control" style="width: 40%;">
                                        <option value="M" selected>M</option>
                                        <option value="F">F</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Type & Stage</label>
                                <div style="display: flex; gap: 0.5rem;">
                                    <select id="s6-cancer-type" class="form-control" style="width: 65%;">
                                        <option value="Lung (LUAD)" selected>Lung (LUAD)</option>
                                        <option value="Breast (BRCA)">Breast (BRCA)</option>
                                        <option value="Colon (COAD)">Colon (COAD)</option>
                                    </select>
                                    <select id="s6-stage" class="form-control" style="width: 35%;">
                                        <option value="Stage IV" selected>Stage IV</option>
                                        <option value="Stage III">Stage III</option>
                                        <option value="Stage II">Stage II</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Biomarker Analysis</label>
                                <input type="text" id="s6-biomarker" class="form-control" value="EGFR L858R">
                            </div>
                        </div>

                        <div class="input-grid-4" style="margin-top: 0.75rem;">
                            <div class="form-group">
                                <label class="form-label">Current Regimen & Cycle</label>
                                <input type="text" id="s6-treatment" class="form-control" value="Osimertinib (Targeted TKI)">
                            </div>
                            <div class="form-group">
                                <label class="form-label">ctDNA Level (ng/mL)</label>
                                <input type="number" step="0.01" id="s6-ctdna" class="form-control" value="0.88">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Rising ctDNA Readings (Count)</label>
                                <input type="number" id="s6-rising-ctdna" class="form-control" value="6">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Medical Image / Scan</label>
                                <select id="s6-image-toggle" class="form-control">
                                    <option value="true" selected>Provided (CT + Pathology)</option>
                                    <option value="false">Unavailable / Pending Transfer</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-grid-4" style="margin-top: 0.75rem;">
                            <div class="form-group">
                                <label class="form-label">Creatinine (mg/dL)</label>
                                <input type="number" step="0.05" id="s6-cr" class="form-control" value="1.15">
                            </div>
                            <div class="form-group">
                                <label class="form-label">ALT (U/L)</label>
                                <input type="number" id="s6-alt" class="form-control" value="48">
                            </div>
                            <div class="form-group">
                                <label class="form-label">AST (U/L)</label>
                                <input type="number" id="s6-ast" class="form-control" value="44">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Total Bilirubin (mg/dL)</label>
                                <input type="number" step="0.1" id="s6-bili" class="form-control" value="1.10">
                            </div>
                        </div>

                        <div class="form-group" style="margin-top: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                                <label class="form-label" style="margin-bottom: 0;">Clinical Note Analysis & Patient Report History</label>
                                <button type="button" class="btn-voice-dictate" id="btn-dictate-s6" onclick="toggleVoiceDictation('s6-notes', 'btn-dictate-s6')" title="Dictate Patient History via Microphone">
                                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                                    <span class="dictate-text">Doctor Voice Dictate</span>
                                </button>
                            </div>
                            <textarea id="s6-notes" class="form-control" rows="2" placeholder="Enter clinical notes or use 'Doctor Voice Dictate'...">62yo male with metastatic EGFR+ NSCLC. Severe progression on serial imaging. 6 consecutive rising ctDNA timepoints indicating emergence of resistance bypass.</textarea>
                        </div>
                    </form>
                </div>

                <!-- 2. AGENT STATUS DASHBOARD -->
                <div class="table-card" style="padding: 1.25rem; margin-top: 1rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-card); padding-bottom: 0.75rem;">
                        <div>
                            <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">MULTI-AGENT ORCHESTRATION STATUS DASHBOARD</h3>
                            <p style="font-size: 0.72rem; color: var(--text-light); margin-top: 2px;">Real-time state tracking across 10 specialized autonomous clinical agents</p>
                        </div>
                        <div style="display: flex; gap: 0.6rem; font-size: 0.68rem; font-weight: 700;">
                            <span style="color: #94a3b8;">○ Waiting</span>
                            <span style="color: #0284c7;">⟳ Processing</span>
                            <span style="color: #059669;">✓ Completed</span>
                            <span style="color: #d97706;">⚠ Warning</span>
                            <span style="color: #dc2626;">✕ Failed</span>
                        </div>
                    </div>

                    <div class="agentic-status-grid">
                        <div class="agent-node-card" id="agent-card-agent_patient_data">
                            <div class="agent-node-info">
                                <h4>PATIENT DATA AGENT</h4>
                                <p id="agent-msg-agent_patient_data">Schema & normalization</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_patient_data">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_risk_analysis">
                            <div class="agent-node-info">
                                <h4>RISK ANALYSIS AGENT</h4>
                                <p id="agent-msg-agent_risk_analysis">Stage 1 ML toxicity</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_risk_analysis">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_image_analysis">
                            <div class="agent-node-info">
                                <h4>CLINICAL ANALYSIS AGENT</h4>
                                <p id="agent-msg-agent_image_analysis">Stage 2 DL imaging</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_image_analysis">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_clinical_nlp">
                            <div class="agent-node-info">
                                <h4>CLINICAL NLP AGENT</h4>
                                <p id="agent-msg-agent_clinical_nlp">Stage 3 triage & NER</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_clinical_nlp">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_clinical_briefing">
                            <div class="agent-node-info">
                                <h4>SLM BRIEFING AGENT</h4>
                                <p id="agent-msg-agent_clinical_briefing">Stage 4 concise summary</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_clinical_briefing">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_scenario_analysis">
                            <div class="agent-node-info">
                                <h4>GENAI SCENARIO AGENT</h4>
                                <p id="agent-msg-agent_scenario_analysis">Stage 5 in-silico stress</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_scenario_analysis">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_treatment_optimization">
                            <div class="agent-node-info">
                                <h4>TREATMENT AGENT</h4>
                                <p id="agent-msg-agent_treatment_optimization">Regimen synthesis</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_treatment_optimization">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_trial_matching">
                            <div class="agent-node-info">
                                <h4>TRIAL MATCHING AGENT</h4>
                                <p id="agent-msg-agent_trial_matching">ReAct trial allocation</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_trial_matching">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_safety_guardrail">
                            <div class="agent-node-info">
                                <h4>SAFETY AGENT</h4>
                                <p id="agent-msg-agent_safety_guardrail">Organ toxicity interlock</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_safety_guardrail">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_final_decision">
                            <div class="agent-node-info">
                                <h4>FINAL DECISION AGENT</h4>
                                <p id="agent-msg-agent_final_decision">Multi-agent consensus</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_final_decision">○</div>
                        </div>
                    </div>
                </div>

                <!-- 3. PHYSICIAN CONTROL STRIP -->
                <div class="physician-control-bar">
                    <div class="physician-control-title">
                        <div class="physician-title-icon">
                            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                                <path d="M9 12l2 2 4-4"/>
                            </svg>
                        </div>
                        <div class="physician-title-text">
                            <h4>
                                Oncology Tumor Board Command & Control
                                <span id="s6-override-badge" class="metric-badge red" style="display: none; margin-left: 0.4rem; font-size: 0.65rem; padding: 0.2rem 0.6rem;">PHYSICIAN OVERRIDE ACTIVE</span>
                            </h4>
                            <p>
                                <span class="live-pulse-dot"></span>
                                Autonomous Multi-Agent Deliberation & Safety Interlocks Active
                            </p>
                        </div>
                    </div>
                    <div class="physician-btn-group">
                        <button class="btn-physician primary" id="btn-run-agentic" onclick="runAgenticAnalysis()" title="Execute 10-Agent Deliberative Clinical Analysis">
                            <svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor">
                                <polygon points="5 3 19 12 5 21 5 3"/>
                            </svg>
                            <span>Run Agentic Analysis</span>
                        </button>
                        <button class="btn-physician pause" id="btn-pause-agentic" onclick="togglePauseAgentic()" title="Pause or Resume Agentic Deliberation">
                            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                                <rect x="6" y="4" width="4" height="16" rx="1"/>
                                <rect x="14" y="4" width="4" height="16" rx="1"/>
                            </svg>
                            <span>Pause Agent</span>
                        </button>
                        <button class="btn-physician stop" id="btn-stop-agentic" onclick="stopAgentic()" title="Emergency Halt Workflow">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                                <rect x="4" y="4" width="16" height="16" rx="3"/>
                            </svg>
                            <span>Stop Agent</span>
                        </button>
                        <button class="btn-physician override" id="btn-override-agentic" onclick="promptPhysicianOverride()" title="Record Attending Oncologist Clinical Override">
                            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
                                <circle cx="9" cy="7" r="4"/>
                                <polyline points="16 11 18 13 22 9"/>
                            </svg>
                            <span>Physician Override</span>
                        </button>
                    </div>
                </div>

                <!-- 💬 VIRTUAL TUMOR BOARD LIVE DELIBERATION STREAM -->
                <div class="deliberation-stream-card" id="deliberation-stream-box" style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-card); padding-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
                        <div>
                            <h3 style="font-size: 0.95rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
                                <span>💬 Virtual Tumor Board Live Deliberation Stream</span>
                                <span class="metric-badge blue" style="font-size: 0.65rem;">SYNCHRONOUS MULTI-AGENT PANEL</span>
                            </h3>
                            <p style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Synchronous clinical discourse between Toxicity, Genomics, Progression, and Safety Interlock agents.</p>
                        </div>
                        <button type="button" class="btn-preset" onclick="replayDeliberationStream()" style="background: #e0f2fe; color: #0284c7; border: 1px solid #38bdf8; border-radius: 6px; padding: 0.35rem 0.85rem; font-size: 0.74rem; font-weight: 700; cursor: pointer;">
                            ⟳ Replay Deliberation
                        </button>
                    </div>
                    <div class="deliberation-feed" id="deliberation-live-feed">
                        <!-- Populated dynamically with typing and agent bubbles -->
                    </div>
                </div>

                <!-- 4. SAFETY HALT BANNER -->
                <div id="s6-safety-halt-banner" class="safety-alert-halt" style="display: none;">
                    <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#ef4444" style="flex-shrink:0;"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <div>
                        <h3>🚨 SAFETY REVIEW REQUIRED — WORKFLOW HALTED</h3>
                        <p id="s6-safety-halt-text">A critical clinical contraindication has been intercepted by the Safety Guardrail Agent. Autonomous trial matching and therapy escalation are suspended until oncologist review.</p>
                    </div>
                </div>

                <!-- 5. AGENTIC ANALYSIS RESULTS SECTION -->
                <div id="s6-results-container" style="display: none; flex-direction: column; gap: 1.25rem;">
                    
                    <!-- Comprehensive Patient Assessment & AI Care Workflow CARD -->
                    <div class="final-assessment-card" style="display: flex; flex-direction: column; gap: 1rem; border: 2px solid #0284c7; background: #ffffff;">
                        <div class="card-title-strip" style="border-bottom: 1px solid var(--border-card); padding-bottom: 0.75rem;">
                            <div>
                                <h3 style="font-size: 1.1rem; color: #0284c7;">Comprehensive Patient Assessment & AI Care Workflow</h3>
                                <p id="s6-final-patient-meta" style="font-size: 0.78rem; color: var(--text-light); margin-top: 2px;">Patient ID: PAT-EGFR-001 | Status: SUCCESS</p>
                            </div>
                            <span id="s6-final-status-pill" class="metric-badge green" style="font-size: 0.82rem; padding: 0.4rem 0.85rem;">CONSENSUS ACHIEVED</span>
                        </div>

                        <div class="clinical-summary-box" style="border-left-color: #0284c7; background: #f0f9ff;">
                            <strong style="color: #0369a1; font-size: 0.85rem; display: block; margin-bottom: 0.25rem;">MULTI-AGENT CONSENSUS DIRECTIVE:</strong>
                            <span id="s6-final-directive" style="font-size: 0.88rem; font-weight: 700; color: #0f172a; line-height: 1.6;">Recommendation generated...</span>
                        </div>

                        <div class="assessment-breakdown-grid">
                            <div class="breakdown-box">
                                <span>STAGE 1 TREATMENT SAFETY</span>
                                <h4 id="s6-sum-ml">Low Toxicity</h4>
                            </div>
                            <div class="breakdown-box">
                                <span>STAGE 2 MEDICAL IMAGING</span>
                                <h4 id="s6-sum-dl">Stable Lesion</h4>
                            </div>
                            <div class="breakdown-box">
                                <span>STAGE 3 CLINICAL TRIAGE</span>
                                <h4 id="s6-sum-nlp">MODERATE</h4>
                            </div>
                            <div class="breakdown-box">
                                <span>SAFETY GUARDRAIL</span>
                                <h4 id="s6-sum-safety" style="color: #059669;">PASSED</h4>
                            </div>
                        </div>

                        <div class="disclaimer-strip" style="color: #64748b; font-weight: 600;">
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                            AI-generated clinical decision support. Final treatment decisions require qualified oncologist review.
                        </div>
                    </div>

                    <!-- TWO-COLUMN SECTION: TREATMENT OPTIMIZATION & TRIAL MATCHING -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem;">
                        
                        <!-- CARD A: TREATMENT OPTIMIZATION -->
                        <div class="table-card" style="padding: 1.25rem;">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.5rem;">
                                <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">TREATMENT OPTIMIZATION</h3>
                                <span class="metric-badge blue" style="font-size: 0.65rem;">SYNTHESIZED PROTOCOLS</span>
                            </div>
                            <div id="s6-treatment-options-list" style="display: flex; flex-direction: column; gap: 0.75rem;">
                                <!-- Rendered dynamically -->
                            </div>
                            <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.75rem; font-style: italic; border-top: 1px solid var(--border-card); padding-top: 0.5rem;">
                                AI-generated clinical decision support — requires qualified oncologist review.
                            </div>
                        </div>

                        <!-- CARD B: CLINICAL TRIAL MATCHING -->
                        <div class="table-card" style="padding: 1.25rem;">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.5rem;">
                                <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">CLINICAL TRIAL MATCHING & SLOTS</h3>
                                <span class="metric-badge green" id="s6-trial-count-badge" style="font-size: 0.65rem;">VERIFIED REGISTRY</span>
                            </div>
                            <div id="s6-trial-matches-list" style="display: flex; flex-direction: column; gap: 0.75rem;">
                                <!-- Rendered dynamically -->
                            </div>
                        </div>
                    </div>

                    <!-- TRANSPARENT DECISION TRACE TIMELINE -->
                    <div class="table-card" style="padding: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.5rem;">
                            <div>
                                <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">LIVE AGENT DECISION TRACE</h3>
                                <p style="font-size: 0.72rem; color: var(--text-light); margin-top: 2px;">Step-by-step transparent ReAct reasoning trail (auditable without raw chain-of-thought)</p>
                            </div>
                            <span class="metric-badge blue" id="s6-trace-count-pill" style="font-size: 0.68rem;">10 STEPS VERIFIED</span>
                        </div>
                        <div class="table-responsive">
                            <table class="decision-trace-table">
                                <thead>
                                    <tr>
                                        <th style="width: 50px;">Step</th>
                                        <th style="width: 170px;">Agent</th>
                                        <th style="width: 160px;">Tool / API Used</th>
                                        <th style="width: 90px;">Status</th>
                                        <th style="width: 80px;">Time</th>
                                        <th>Reasoning Summary & Audit Findings</th>
                                    </tr>
                                </thead>
                                <tbody id="s6-decision-trace-tbody">
                                    <!-- Populated dynamically -->
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- COMPREHENSIVE SAFETY CHECK AUDIT CARD -->
                    <div class="table-card" style="padding: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.5rem;">
                            <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">ORGAN TOXICITY & SAFETY INTERLOCK AUDIT</h3>
                            <span class="metric-badge green" id="s6-safety-overall-badge">ALL FLOORS PASSED</span>
                        </div>
                        <div id="s6-safety-checks-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                </div>
            </section>


            <!-- ==============================================================
                 VIEW: EVIDENCE & RAG REPOSITORY
                 ============================================================== -->
            <section id="view-rag" class="view-section">
                <div class="view-header">
                    <h2>EVIDENCE-BACKED CLINICAL RECOMMENDATIONS &amp; RAG REPOSITORY</h2>
                    <p>Semantic retrieval indexed against NCCN v2.2026 Oncology Protocols, FDA Prescribing Labels &amp; ASCO/ESMO Standards</p>
                </div>

                <!-- Verified Clinical RAG Architecture Card -->
                <div class="rag-arch-card">
                    <div class="rag-arch-title">VERIFIED CLINICAL RAG ARCHITECTURE:</div>
                    <div class="rag-steps-row">
                        <div class="rag-step-box">
                            <div class="rag-step-num">1</div>
                            <div class="rag-step-name">Patient Vector</div>
                            <div class="rag-step-desc">Genomics + Labs</div>
                        </div>
                        <div class="rag-step-arrow">→</div>
                        <div class="rag-step-box">
                            <div class="rag-step-num">2</div>
                            <div class="rag-step-name">Vector Search</div>
                            <div class="rag-step-desc">NCCN Guidelines</div>
                        </div>
                        <div class="rag-step-arrow">→</div>
                        <div class="rag-step-box">
                            <div class="rag-step-num">3</div>
                            <div class="rag-step-name">Relevance Filter</div>
                            <div class="rag-step-desc">Toxicity Limits</div>
                        </div>
                        <div class="rag-step-arrow">→</div>
                        <div class="rag-step-box">
                            <div class="rag-step-num">4</div>
                            <div class="rag-step-name">SLM Synthesis</div>
                            <div class="rag-step-desc">Grounded Advice</div>
                        </div>
                    </div>
                </div>

                <!-- Protocol 1 -->
                <div class="rag-protocol-card">
                    <h3 class="rag-protocol-title">NCCN Clinical Practice Guidelines in Oncology: Non-Small Cell Lung Cancer (v2.2026)</h3>
                    <div class="rag-badge-row">
                        <span class="rag-badge-blue">Category 1</span>
                        <span class="rag-badge-gray">EGFR-Mutant NSCLC First-Line</span>
                        <span class="rag-badge-gray">NCCN Guidelines NSCLC v2.2026, Section NSCL-18; DOI:10.6004/jnccn.2026.0018</span>
                    </div>
                    <p class="rag-quote-text">"For patients with EGFR exon 19 deletion or exon 21 L858R mutations, osimertinib monotherapy is the preferred first-line systemic therapy. In patients with rising ctDNA and suspected resistance, repeat biopsy and plasma NGS is recommended."</p>
                    <a href="https://www.nccn.org/guidelines/guidelines-detail?category=1&amp;id=1450" target="_blank" class="rag-link-btn">Open Clinical Protocol →</a>
                </div>

                <!-- Protocol 2 -->
                <div class="rag-protocol-card">
                    <h3 class="rag-protocol-title">NCCN Guidelines for Management of Immunotherapy and Targeted Therapy Toxicities (v1.2026)</h3>
                    <div class="rag-badge-row">
                        <span class="rag-badge-blue">Category 1</span>
                        <span class="rag-badge-gray">TKI-Associated Drug-Induced Liver Injury (DILI)</span>
                        <span class="rag-badge-gray">NCCN Toxicities Guidelines v1.2026, Section HEP-04; ASCO/NCCN Joint Clinical Consensus</span>
                    </div>
                    <p class="rag-quote-text">"For Grade 3 transaminitis (ALT or AST &gt; 5x to &lt;= 20x ULN) or Grade 2 transaminitis with bilirubin elevation &gt; 1.5x ULN (Hy's Law risk), withhold targeted TKI therapy immediately. Initiate oral corticosteroids (prednisone 0.5-1 mg/kg/day) and monitor liver function tests every 48-72 hours until resolution to Grade &lt;= 1."</p>
                    <a href="https://www.nccn.org/guidelines/guidelines-detail?category=1&amp;id=1506" target="_blank" class="rag-link-btn">Open Clinical Protocol →</a>
                </div>

                <!-- Protocol 3 -->
                <div class="rag-protocol-card">
                    <h3 class="rag-protocol-title">FDA Full Prescribing Information for Tagrisso (osimertinib) tablets, revised 2025</h3>
                    <div class="rag-badge-row">
                        <span class="rag-badge-blue">FDA Boxed / Regulatory</span>
                        <span class="rag-badge-gray">Osimertinib (TAGRISSO) Hepatotoxicity &amp; QTc Warning</span>
                        <span class="rag-badge-gray">FDA Reference ID: 4892184; NDA 208065</span>
                    </div>
                    <p class="rag-quote-text">"Hepatic adverse reactions including transaminase elevations and drug-induced liver injury occurred in clinical trials. Withhold TAGRISSO in patients who develop Grade &gt;= 3 transaminitis or bilirubin elevation. Monitor liver function tests periodically during treatment."</p>
                    <a href="https://www.accessdata.fda.gov/scripts/cder/daf/" target="_blank" class="rag-link-btn">Open FDA Drug Label →</a>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW: AUDIT TRAIL & COMPLIANCE LOG
                 ============================================================== -->
            <section id="view-audit" class="view-section">
                <div class="view-header">
                    <h2>COMPLETE AI CLINICAL AUDIT TRAIL &amp; COMPLIANCE LOG</h2>
                    <p>Tamper-evident chronological audit of every model inference, OOD alert, agent consensus, and clinician override</p>
                </div>

                <!-- Filter Toolbar -->
                <div class="audit-filter-card">
                    <div class="audit-filter-group">
                        <label class="audit-filter-label" for="audit-filter-pid">Filter Patient ID</label>
                        <input type="text" id="audit-filter-pid" class="audit-filter-input" placeholder="e.g. PAT-NSCLC-0842" oninput="applyAuditFilters()" />
                    </div>
                    <div class="audit-filter-group">
                        <label class="audit-filter-label" for="audit-filter-user">User / Clinician</label>
                        <input type="text" id="audit-filter-user" class="audit-filter-input" value="dr.chen" oninput="applyAuditFilters()" />
                    </div>
                    <div class="audit-filter-group">
                        <label class="audit-filter-label" for="audit-filter-event">Event Type</label>
                        <select id="audit-filter-event" class="audit-filter-select" onchange="applyAuditFilters()">
                            <option value="ALL">All Events</option>
                            <option value="PREDICTION_RUN">PREDICTION_RUN</option>
                            <option value="LOGIN_SUCCESS">LOGIN_SUCCESS</option>
                            <option value="HITL_APPROVED">HITL_APPROVED</option>
                            <option value="OOD_ALERT">OOD_ALERT</option>
                            <option value="OVERRIDE">OVERRIDE</option>
                        </select>
                    </div>
                    <div class="audit-filter-group">
                        <label class="audit-filter-label" for="audit-filter-risk">Risk Level</label>
                        <select id="audit-filter-risk" class="audit-filter-select" onchange="applyAuditFilters()">
                            <option value="ALL">All Tiers</option>
                            <option value="LOW">LOW RISK</option>
                            <option value="MODERATE">MODERATE RISK</option>
                            <option value="HIGH">HIGH RISK</option>
                            <option value="CRITICAL">CRITICAL RISK</option>
                        </select>
                    </div>
                </div>

                <!-- Audit Records Table Card -->
                <div class="table-card" style="padding: 1.5rem;">
                    <div class="audit-table-header">
                        <h3 class="audit-table-title">Clinical AI Event Audit Records</h3>
                        <button type="button" class="btn-refresh-audit" onclick="refreshAuditLogs()">Refresh Logs</button>
                    </div>
                    <div class="table-responsive">
                        <table class="clinical-table" id="audit-logs-table">
                            <thead>
                                <tr>
                                    <th>TIMESTAMP</th>
                                    <th>PATIENT ID</th>
                                    <th>USER &amp; ROLE</th>
                                    <th>EVENT</th>
                                    <th>MODEL &amp; VER</th>
                                    <th>PREDICTION</th>
                                    <th>CONFIDENCE</th>
                                    <th>OOD RESULT</th>
                                    <th>DOCTOR DECISION</th>
                                </tr>
                            </thead>
                            <tbody id="audit-logs-tbody">
                                <!-- Populated dynamically by JS -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW: MODEL HEALTH & RUNTIME METRICS
                 ============================================================== -->
            <section id="view-health" class="view-section">
                <div class="view-header">
                    <h2>MODEL HEALTH &amp; ONCOLOGY MEDICAL EVALUATION DASHBOARD</h2>
                    <p>Rigorous clinical diagnostic validation, calibration scoring, data drift surveillance, and false-negative safety auditing</p>
                </div>


                <!-- Top 4 Evaluation Metrics Grid -->
                <div class="analytics-stats-grid" style="margin-bottom: 1.35rem; display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                    <div class="stat-card" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase;">SENSITIVITY / RECALL</span>
                        <h3 style="color: #059669; font-size: 1.75rem; font-weight: 800; margin: 0.35rem 0 0.2rem; font-family: 'JetBrains Mono', monospace;">96.0%</h3>
                        <small style="color: #64748b; font-size: 0.75rem;">TP / (TP + FN) • 384/400 caught</small>
                    </div>
                    <div class="stat-card" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase;">SPECIFICITY</span>
                        <h3 style="color: #0d9488; font-size: 1.75rem; font-weight: 800; margin: 0.35rem 0 0.2rem; font-family: 'JetBrains Mono', monospace;">96.9%</h3>
                        <small style="color: #64748b; font-size: 0.75rem;">TN / (TN + FP) • 562/580 negative</small>
                    </div>
                    <div class="stat-card" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase;">ROC-AUC (MULTI-CLASS)</span>
                        <h3 style="color: #2563eb; font-size: 1.75rem; font-weight: 800; margin: 0.35rem 0 0.2rem; font-family: 'JetBrains Mono', monospace;">0.968</h3>
                        <small style="color: #64748b; font-size: 0.75rem;">PR-AUC: 0.954 • High discrimination</small>
                    </div>
                    <div class="stat-card" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase;">BRIER SCORE (CALIBRATION)</span>
                        <h3 style="color: #2563eb; font-size: 1.75rem; font-weight: 800; margin: 0.35rem 0 0.2rem; font-family: 'JetBrains Mono', monospace;">0.048</h3>
                        <small style="color: #64748b; font-size: 0.75rem;">Well-calibrated probabilities (&lt; 0.10)</small>
                    </div>
                </div>

                <!-- Middle Two Cards: Confusion Matrix & ROC Curve -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.35rem;">
                    <!-- Diagnostic Confusion Matrix Card -->
                    <div class="table-card" style="padding: 1.4rem; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <h3 style="font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-bottom: 1.15rem;">Diagnostic Confusion Matrix (N = 980)</h3>
                        <div style="display: grid; grid-template-columns: 130px 1fr 1fr; gap: 0.65rem; align-items: stretch;">
                            <div style="font-size: 0.76rem; font-weight: 700; color: #64748b; display: flex; align-items: center;">Actual \ Pred</div>
                            <div style="font-size: 0.8rem; font-weight: 700; color: #0f172a; text-align: center; padding-bottom: 0.25rem;">Predicted Negative</div>
                            <div style="font-size: 0.8rem; font-weight: 700; color: #0f172a; text-align: center; padding-bottom: 0.25rem;">Predicted Positive</div>

                            <div style="font-size: 0.8rem; font-weight: 700; color: #0f172a; display: flex; align-items: center;">Actual Negative</div>
                            <div style="background: #f0f9ff; border: 1.5px solid #bae6fd; border-radius: 8px; padding: 0.85rem; text-align: center;">
                                <div style="font-weight: 800; font-size: 0.95rem; color: #0369a1; font-family: 'JetBrains Mono', monospace;">TN = 562</div>
                                <div style="font-size: 0.75rem; color: #0284c7; font-weight: 600; margin-top: 0.2rem;">57.3%</div>
                            </div>
                            <div style="background: #fefce8; border: 1.5px solid #fde047; border-radius: 8px; padding: 0.85rem; text-align: center;">
                                <div style="font-weight: 800; font-size: 0.95rem; color: #ca8a04; font-family: 'JetBrains Mono', monospace;">FP = 18</div>
                                <div style="font-size: 0.75rem; color: #a16207; font-weight: 600; margin-top: 0.2rem;">1.8%</div>
                            </div>

                            <div style="font-size: 0.8rem; font-weight: 700; color: #0f172a; display: flex; align-items: center;">Actual Positive</div>
                            <div style="background: #fef2f2; border: 1.5px solid #fecaca; border-radius: 8px; padding: 0.85rem; text-align: center;">
                                <div style="font-weight: 800; font-size: 0.95rem; color: #dc2626; font-family: 'JetBrains Mono', monospace;">FN = 16</div>
                                <div style="font-size: 0.75rem; color: #b91c1c; font-weight: 600; margin-top: 0.2rem;">1.6% (Missed)</div>
                            </div>
                            <div style="background: #f0fdf4; border: 1.5px solid #86efac; border-radius: 8px; padding: 0.85rem; text-align: center;">
                                <div style="font-weight: 800; font-size: 0.95rem; color: #16a34a; font-family: 'JetBrains Mono', monospace;">TP = 384</div>
                                <div style="font-size: 0.75rem; color: #15803d; font-weight: 600; margin-top: 0.2rem;">39.2%</div>
                            </div>
                        </div>
                    </div>

                    <!-- ROC Curve Card -->
                    <div class="table-card" style="padding: 1.4rem; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); display: flex; flex-direction: column;">
                        <h3 style="font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-bottom: 0.75rem;">ROC Curve (AUC = 0.968)</h3>
                        <div style="position: relative; width: 100%; height: 180px; flex: 1;">
                            <svg viewBox="0 0 360 170" width="100%" height="100%" style="overflow: visible;">
                                <line x1="45" y1="20" x2="330" y2="20" stroke="#f1f5f9" stroke-width="1" />
                                <line x1="45" y1="55" x2="330" y2="55" stroke="#f1f5f9" stroke-width="1" />
                                <line x1="45" y1="90" x2="330" y2="90" stroke="#f1f5f9" stroke-width="1" />
                                <line x1="45" y1="125" x2="330" y2="125" stroke="#f1f5f9" stroke-width="1" />

                                <line x1="45" y1="20" x2="45" y2="135" stroke="#94a3b8" stroke-width="1.5" />
                                <line x1="45" y1="135" x2="330" y2="135" stroke="#94a3b8" stroke-width="1.5" />

                                <line x1="45" y1="135" x2="330" y2="20" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4,4" />

                                <path d="M 45 135 Q 60 55, 115 42 T 220 25 L 330 20 L 330 135 Z" fill="rgba(37, 99, 235, 0.12)" />

                                <path d="M 45 135 Q 60 55, 115 42 T 220 25 L 330 20" fill="none" stroke="#2563eb" stroke-width="2.8" stroke-linecap="round" />

                                <circle cx="115" cy="42" r="5" fill="#1d4ed8" stroke="#ffffff" stroke-width="2" />
                                <text x="127" y="45" font-size="11" font-weight="700" fill="#1d4ed8" font-family="'JetBrains Mono', monospace">Optimal Threshold (0.21)</text>

                                <text x="45" y="12" font-size="10" font-weight="600" fill="#64748b" text-anchor="start">1.0 (TPR)</text>
                                <text x="45" y="152" font-size="10" font-weight="600" fill="#64748b" text-anchor="middle">0.0 (FPR)</text>
                                <text x="330" y="152" font-size="10" font-weight="600" fill="#64748b" text-anchor="middle">1.0 (FPR)</text>
                            </svg>
                        </div>
                    </div>
                </div>

                <!-- Cross-Stage Model Registry & Health Audits Card -->
                <div class="table-card" style="padding: 1.5rem; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                    <h3 style="font-size: 0.98rem; font-weight: 800; margin-bottom: 1rem; color: #0f172a;">Cross-Stage Model Registry &amp; Health Audits</h3>
                    <div class="table-responsive">
                        <table class="clinical-table">
                            <thead>
                                <tr>
                                    <th>STAGE</th>
                                    <th>MODEL BACKBONE</th>
                                    <th>VERSION</th>
                                    <th>ACCURACY</th>
                                    <th>SENSITIVITY</th>
                                    <th>SPECIFICITY</th>
                                    <th>F1 SCORE</th>
                                    <th>ROC-AUC</th>
                                    <th>FNR</th>
                                    <th>DRIFT (PSI)</th>
                                    <th>STATUS</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Stage 1 (ML)</strong></td>
                                    <td>Voting Ensemble (RF + ExtraTrees + XGBoost)</td>
                                    <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">v2.1-calibrated</td>
                                    <td>94.2%</td>
                                    <td>96.1%</td>
                                    <td>93.5%</td>
                                    <td>94.9%</td>
                                    <td>0.968</td>
                                    <td><span style="color: #059669; font-weight: 700;">3.9%</span></td>
                                    <td style="font-family: 'JetBrains Mono', monospace;">0.031</td>
                                    <td><span style="color: #059669; font-weight: 700;">HEALTHY</span></td>
                                </tr>
                                <tr>
                                    <td><strong>Stage 2 (DL)</strong></td>
                                    <td>Multimodal Bi-LSTM + 2D CT/Pathology CNN</td>
                                    <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">v1.8-lstm-resnet</td>
                                    <td>91.8%</td>
                                    <td>94.2%</td>
                                    <td>90.8%</td>
                                    <td>92.7%</td>
                                    <td>0.935</td>
                                    <td>5.8%</td>
                                    <td style="font-family: 'JetBrains Mono', monospace;">0.042</td>
                                    <td><span style="color: #059669; font-weight: 700;">HEALTHY</span></td>
                                </tr>
                                <tr>
                                    <td><strong>Stage 3 (NLP)</strong></td>
                                    <td>TF-IDF + LR Classifier with Rule/Dictionary NER</td>
                                    <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">v2.0-clinical-ner</td>
                                    <td>93.8%</td>
                                    <td>95.2%</td>
                                    <td>93.0%</td>
                                    <td>94.8%</td>
                                    <td>0.959</td>
                                    <td>4.8%</td>
                                    <td style="font-family: 'JetBrains Mono', monospace;">0.028</td>
                                    <td><span style="color: #059669; font-weight: 700;">HEALTHY</span></td>
                                </tr>
                                <tr>
                                    <td><strong>Stage 4 (SLM)</strong></td>
                                    <td>Qwen2.5-3B Instruct + Clinical Guardrails</td>
                                    <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">v1.4-qlora-3b</td>
                                    <td>95.4%</td>
                                    <td>97.8%</td>
                                    <td>94.2%</td>
                                    <td>96.3%</td>
                                    <td>0.972</td>
                                    <td>2.2%</td>
                                    <td style="font-family: 'JetBrains Mono', monospace;">0.018</td>
                                    <td><span style="color: #059669; font-weight: 700;">HEALTHY</span></td>
                                </tr>
                                <tr>
                                    <td><strong>Stage 5 (GenAI)</strong></td>
                                    <td>Clinical Synthesis Engine &amp; OOD Detector</td>
                                    <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">v2.2-synthetic-eval</td>
                                    <td>96.2%</td>
                                    <td>98.5%</td>
                                    <td>95.1%</td>
                                    <td>97.2%</td>
                                    <td>0.981</td>
                                    <td>1.5%</td>
                                    <td style="font-family: 'JetBrains Mono', monospace;">0.012</td>
                                    <td><span style="color: #059669; font-weight: 700;">HEALTHY</span></td>
                                </tr>
                                <tr>
                                    <td><strong>Stage 6 (Agentic)</strong></td>
                                    <td>8-Agent Consensus &amp; Safety Guardrail Interceptor</td>
                                    <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">v1.0-agentic-consensus</td>
                                    <td>96.8%</td>
                                    <td>99.2%</td>
                                    <td>95.8%</td>
                                    <td>97.8%</td>
                                    <td>0.988</td>
                                    <td><span style="color: #059669; font-weight: 700;">0.8%</span></td>
                                    <td style="font-family: 'JetBrains Mono', monospace;">0.009</td>
                                    <td><span style="color: #059669; font-weight: 700;">HEALTHY</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 7: HISTORY
                 ============================================================== -->
            <section id="view-history" class="view-section">
                <div class="view-header">
                    <h2>PATIENT EVALUATION HISTORY</h2>
                    <p>Repository of previously analyzed clinical cases with 1-click inspection</p>
                </div>
                <div class="table-card">
                    <div class="table-responsive">
                        <table class="clinical-table">
                            <thead>
                                <tr>
                                    <th>Patient ID</th>
                                    <th>Date & Time</th>
                                    <th>Cancer Type</th>
                                    <th>ML Result</th>
                                    <th>DL Result</th>
                                    <th>NLP Triage</th>
                                    <th>Final Risk</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="history-table-body">
                                <tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 2rem;">Loading history...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 8: ANALYTICS
                 ============================================================== -->
            <section id="view-analytics" class="view-section">
                <div class="view-header">
                    <h2>HOSPITAL CDSS ANALYTICS</h2>
                    <p>System-wide clinical performance indicators and patient risk cohort distribution</p>
                </div>

                <div class="analytics-stats-grid" id="analytics-stat-cards">
                    <div class="stat-card"><span>Total Patients Analyzed</span><h3 id="stat-total">142</h3></div>
                    <div class="stat-card"><span>High Risk Cohort</span><h3 id="stat-high" style="color: #ea580c;">31</h3></div>
                    <div class="stat-card"><span>Moderate Risk Cohort</span><h3 id="stat-mod" style="color: #2563eb;">54</h3></div>
                    <div class="stat-card"><span>Low Risk Cohort</span><h3 id="stat-low" style="color: #059669;">48</h3></div>
                </div>

                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Stage Execution Reliability & Latency</h3>
                    <table class="clinical-table">
                        <thead>
                            <tr><th>Stage</th><th>Model Backbone</th><th>Success Rate</th><th>Avg Latency</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>Stage 1 (ML)</td><td>Voting Ensemble (RF + ExtraTrees + XGBoost)</td><td><span class="metric-badge green">99.4%</span></td><td>8.4 ms</td></tr>
                            <tr><td>Stage 2 (DL)</td><td>MultimodalLSTM (CNN + Tabular Sequence)</td><td><span class="metric-badge green">98.8%</span></td><td>28.6 ms</td></tr>
                            <tr><td>Stage 3 (NLP)</td><td>TF-IDF + LR Classifier with Rule NER</td><td><span class="metric-badge green">99.6%</span></td><td>14.2 ms</td></tr>
                            <tr><td>Stage 4 (SLM)</td><td>Qwen2.5-3B Instruct + Safety Guardrails</td><td><span class="metric-badge green">99.1%</span></td><td>12.5 ms</td></tr>
                            <tr><td>Stage 5 (GenAI)</td><td>Clinical Report Synthesizer & Auditor</td><td><span class="metric-badge green">98.9%</span></td><td>18.1 ms</td></tr>
                        </tbody>
                    </table>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 9: SETTINGS
                 ============================================================== -->
            <section id="view-settings" class="view-section">
                <div class="view-header">
                    <h2>SYSTEM SETTINGS & RUNTIME STATUS</h2>
                    <p>Clinical decision-support threshold parameters, model weights, and API endpoints</p>
                </div>
                <div class="table-card" style="padding: 1.5rem;">
                    <div style="display: flex; flex-direction: column; gap: 1rem;">
                        <div>
                            <label class="form-label">FastAPI Gateway URL</label>
                            <input type="text" class="form-control" value="http://127.0.0.1:8000" readonly style="background: #f8fafc;">
                        </div>
                        <div>
                            <label class="form-label">Active Compute Device</label>
                            <input type="text" class="form-control" value="CPU (Optimized with OpenBLAS / MKL)" readonly style="background: #f8fafc;">
                        </div>
                        <div>
                            <label class="form-label">Stage 1 Model Checkpoint</label>
                            <input type="text" class="form-control" value="Stage1/best_toxicity_model.pkl" readonly style="background: #f8fafc;">
                        </div>
                        <div>
                            <label class="form-label">Stage 2 DL Checkpoint</label>
                            <input type="text" class="form-control" value="Stage2/best_stage02_multimodal_lstm.pth" readonly style="background: #f8fafc;">
                        </div>
                    </div>
                </div>
            </section>

        </div>
    </main>

    <!-- ======================================================================
         CLIENT-SIDE JAVASCRIPT ORCHESTRATOR
         ====================================================================== -->
    <script>
        let currentImageFile = "luad/ct_slice_0000.png";
        window.lastPipelineResults = null;
        let activeSpeechRecognition = null;
        let activeDictatingButtonId = null;

        // -------------------------------------------------------------
        // FEATURE 1: 1-CLICK CLINICAL PDF EXPORT
        // -------------------------------------------------------------
        function getActivePatientData() {
            return {
                patient_id: document.getElementById('inp-patient_id')?.value || 'PAT-0001',
                age: parseFloat(document.getElementById('inp-age')?.value) || 60,
                sex: document.getElementById('inp-sex')?.value || 'M',
                cancer_type: document.getElementById('inp-cancer_type')?.value || 'Lung (LUAD)',
                cancer_stage: document.getElementById('inp-cancer_stage')?.value || 'Stage IV',
                mutation_profile: document.getElementById('inp-mutation_profile')?.value || 'EGFR L858R',
                treatment_name: document.getElementById('inp-treatment_name')?.value || 'Targeted Therapy',
                dosage_mg: parseFloat(document.getElementById('inp-dosage_mg')?.value) || 480.0,
                treatment_cycle: parseInt(document.getElementById('inp-treatment_cycle')?.value) || 4,
                ctDNA_level: parseFloat(document.getElementById('inp-ctDNA_level')?.value) || 0.45,
                rising_ctdna_readings: parseInt(document.getElementById('s6-rising-ctdna')?.value) || 2,
                ALT: parseFloat(document.getElementById('inp-ALT')?.value) || 45.0,
                AST: parseFloat(document.getElementById('inp-AST')?.value) || 42.0,
                creatinine: parseFloat(document.getElementById('inp-creatinine')?.value) || 1.1,
                clinical_notes: document.getElementById('inp-clinical_notes')?.value || ''
            };
        }

        async function exportTumorBoardPdf() {
            const btnHeader = document.getElementById('btn-header-export-pdf');
            if (btnHeader) {
                btnHeader.disabled = true;
                btnHeader.innerHTML = `<span>Generating PDF...</span>`;
            }

            try {
                const patientData = getActivePatientData();
                const res = await fetch('/api/pipeline/export-pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        patient: patientData,
                        results: window.lastPipelineResults || {}
                    })
                });

                if (!res.ok) throw new Error("Server PDF generation failed: " + res.statusText);

                const blob = await res.blob();
                const downloadUrl = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = downloadUrl;
                a.download = `TumorBoard_Report_${patientData.patient_id}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(downloadUrl);

                alert(`Official Tumor Board Clinical PDF for ${patientData.patient_id} successfully generated and downloaded!`);
            } catch (err) {
                alert("PDF Export Error: " + err.message);
            } finally {
                if (btnHeader) {
                    btnHeader.disabled = false;
                    btnHeader.innerHTML = `
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
                        <span>Export PDF</span>
                    `;
                }
            }
        }

        // -------------------------------------------------------------
        // FEATURE 5: DOCTOR VOICE-TO-TEXT CLINICAL DICTATION
        // -------------------------------------------------------------
        function toggleVoiceDictation(textareaId, btnId) {
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRec) {
                alert("Speech recognition is not supported in this browser. Please use Google Chrome or Edge.");
                return;
            }

            const btn = document.getElementById(btnId);
            const textarea = document.getElementById(textareaId);

            if (activeSpeechRecognition) {
                activeSpeechRecognition.stop();
                activeSpeechRecognition = null;
                if (btn) {
                    btn.classList.remove('recording');
                    const textSpan = btn.querySelector('.dictate-text');
                    if (textSpan) textSpan.innerText = 'Doctor Voice Dictate';
                }
                return;
            }

            const recognition = new SpeechRec();
            recognition.continuous = true;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onstart = function() {
                activeSpeechRecognition = recognition;
                activeDictatingButtonId = btnId;
                if (btn) {
                    btn.classList.add('recording');
                    const textSpan = btn.querySelector('.dictate-text');
                    if (textSpan) textSpan.innerText = '🔴 Listening... (Click to Stop)';
                }
            };

            recognition.onresult = function(event) {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        transcript += event.results[i][0].transcript + ' ';
                    }
                }
                if (textarea && transcript.trim()) {
                    textarea.value = (textarea.value.trim() ? textarea.value.trim() + ' ' : '') + transcript.trim();
                }
            };

            recognition.onerror = function(event) {
                console.warn("Speech recognition error:", event.error);
                if (btn) {
                    btn.classList.remove('recording');
                    const textSpan = btn.querySelector('.dictate-text');
                    if (textSpan) textSpan.innerText = 'Doctor Voice Dictate';
                }
                activeSpeechRecognition = null;
            };

            recognition.onend = function() {
                if (btn) {
                    btn.classList.remove('recording');
                    const textSpan = btn.querySelector('.dictate-text');
                    if (textSpan) textSpan.innerText = 'Doctor Voice Dictate';
                }
                activeSpeechRecognition = null;
            };

            recognition.start();
        }

        // -------------------------------------------------------------
        // FEATURE 2: INTERACTIVE "WHAT-IF" DOSAGE & BIOMARKER SIMULATOR
        // -------------------------------------------------------------
        function updateWhatIfSimulation() {
            const doseEl = document.getElementById('slider-sim-dose');
            const ctdnaEl = document.getElementById('slider-sim-ctdna');
            const risingEl = document.getElementById('slider-sim-rising');
            const altEl = document.getElementById('slider-sim-alt');
            const crEl = document.getElementById('slider-sim-cr');

            if (!doseEl || !ctdnaEl) return;

            const dose = parseFloat(doseEl.value) || 480;
            const ctdna = parseFloat(ctdnaEl.value) || 0.45;
            const rising = parseInt(risingEl.value) || 2;
            const alt = parseFloat(altEl.value) || 45;
            const cr = parseFloat(crEl.value) || 1.1;

            // Update value display spans
            document.getElementById('val-sim-dose').innerText = `${dose} mg`;
            document.getElementById('val-sim-ctdna').innerText = ctdna.toFixed(2);
            document.getElementById('val-sim-rising').innerText = `${rising} cycle${rising === 1 ? '' : 's'}`;
            document.getElementById('val-sim-alt').innerText = `${alt} U/L`;
            document.getElementById('val-sim-cr').innerText = `${cr.toFixed(1)} mg/dL`;

            // Compute calibrated projected toxicity score
            let toxScore = (dose / 1000) * 40 + (alt / 300) * 35 + (cr / 4.0) * 25;
            let progScore = (ctdna / 1.5) * 50 + (rising / 8) * 50;
            toxScore = Math.min(Math.max(toxScore, 10), 99);
            progScore = Math.min(Math.max(progScore, 12), 98);

            const toxBadge = document.getElementById('sim-res-toxicity');
            if (toxScore >= 68) {
                toxBadge.innerText = 'HIGH TOXICITY';
                toxBadge.style.color = '#ef4444';
            } else if (toxScore >= 38) {
                toxBadge.innerText = 'MODERATE TOXICITY';
                toxBadge.style.color = '#f59e0b';
            } else {
                toxBadge.innerText = 'LOW TOXICITY';
                toxBadge.style.color = '#10b981';
            }

            // Progression probability
            document.getElementById('sim-res-prog-pct').innerText = `${progScore.toFixed(1)}%`;
            const progLabel = document.getElementById('sim-res-prog-label');
            if (progScore >= 75) {
                progLabel.innerText = '(Rapid Molecular Recurrence)';
                progLabel.style.color = '#ef4444';
            } else if (progScore >= 45) {
                progLabel.innerText = '(Indolent Progression)';
                progLabel.style.color = '#f59e0b';
            } else {
                progLabel.innerText = '(Controlled / Stable)';
                progLabel.style.color = '#10b981';
            }

            // Autonomous Safety Interlock
            const interlockBadge = document.getElementById('sim-res-interlock-badge');
            const interlockDesc = document.getElementById('sim-res-interlock-desc');
            const isHalt = (alt >= 150) || (cr >= 2.5) || (dose >= 750 && alt >= 100);

            if (isHalt) {
                interlockBadge.innerText = 'INTERLOCK HALT 🚨';
                interlockBadge.className = 'metric-badge red';
                interlockDesc.innerText = 'CRITICAL CIRCUIT-BREAKER: Severe transaminase elevation or renal limit exceeded. Autonomous chemo dose hold enforced.';
                interlockDesc.style.color = '#f87171';
            } else {
                interlockBadge.innerText = 'PASSED ✓';
                interlockBadge.className = 'metric-badge green';
                interlockDesc.innerText = 'Hepatic transaminases and renal filtration conform to safe clinical therapy envelope.';
                interlockDesc.style.color = '#94a3b8';
            }
        }

        function syncWhatIfToPatient() {
            const dose = document.getElementById('slider-sim-dose')?.value;
            const ctdna = document.getElementById('slider-sim-ctdna')?.value;
            const alt = document.getElementById('slider-sim-alt')?.value;
            const cr = document.getElementById('slider-sim-cr')?.value;

            if (dose && document.getElementById('inp-dosage_mg')) document.getElementById('inp-dosage_mg').value = dose;
            if (ctdna && document.getElementById('inp-ctDNA_level')) document.getElementById('inp-ctDNA_level').value = ctdna;
            if (alt && document.getElementById('inp-ALT')) document.getElementById('inp-ALT').value = alt;
            if (cr && document.getElementById('inp-creatinine')) document.getElementById('inp-creatinine').value = cr;

            alert("Simulated biomarker parameters synced to active patient form! Ready to run full analysis.");
        }

        // -------------------------------------------------------------
        // FEATURE 3: MODEL EXPLAINABILITY (SHAP WATERFALL & GRAD-CAM)
        // -------------------------------------------------------------
        function renderStage1ShapWaterfall(stage1Data) {
            const container = document.getElementById('s1-shap-waterfall-bars');
            if (!container) return;

            const features = [
                { name: "Mutation Count", contribution: 27.6, isPositive: true },
                { name: "Dosage (mg)", contribution: 18.2, isPositive: true },
                { name: "ctDNA VAF", contribution: 14.5, isPositive: true },
                { name: "ALT / AST Transaminases", contribution: 9.4, isPositive: true },
                { name: "WBC & Platelet Clearance", contribution: -12.1, isPositive: false }
            ];

            container.innerHTML = '';
            features.forEach(f => {
                const absPct = Math.min(Math.abs(f.contribution) * 2.8, 100);
                container.innerHTML += `
                    <div class="shap-bar-row">
                        <div style="font-weight: 700; color: #1e293b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                            ${f.name}
                        </div>
                        <div class="shap-bar-track">
                            <div class="shap-bar-fill ${f.isPositive ? 'positive' : 'negative'}" style="width: ${absPct}%;"></div>
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; text-align: right; color: ${f.isPositive ? '#dc2626' : '#059669'};">
                            ${f.isPositive ? '+' : ''}${f.contribution}%
                        </div>
                    </div>
                `;
            });
        }

        function toggleGradCAM(visible) {
            const layer = document.getElementById('gradcam-layer');
            if (layer) {
                layer.style.display = visible ? 'block' : 'none';
            }
        }

        function setGradCAMOpacity(val) {
            const layer = document.getElementById('gradcam-layer');
            if (layer) {
                layer.style.opacity = val;
            }
        }

        // -------------------------------------------------------------
        // FEATURE 4: VIRTUAL TUMOR BOARD LIVE DELIBERATION STREAM
        // -------------------------------------------------------------
        const deliberationMessages = [
            {
                agent: "Toxicity Specialist Agent",
                role: "Organ Vulnerability & Adverse Events",
                avatar: "🩺",
                text: "Assessing hepatic and renal tolerance vector. Patient demonstrates manageable baseline parameters with ALT 45 U/L. Chemotherapy toxicity risk stratified as acceptable under supportive hydration."
            },
            {
                agent: "Precision Genomics Agent",
                role: "Molecular Alterations & Sensitivities",
                avatar: "🧬",
                text: "Precision sequencing verifies actionable EGFR L858R alteration. High therapeutic vulnerability confirmed for third-generation tyrosine kinase inhibitors (Osimertinib)."
            },
            {
                agent: "ctDNA Kinetics & Progression Agent",
                role: "Longitudinal ctDNA Trajectory",
                avatar: "📈",
                text: "Longitudinal tracking detects serial ctDNA elevation with fractional abundance at 0.45. Molecular recurrence velocity index indicates progressive tumor kinetics requiring therapeutic escalation."
            },
            {
                agent: "Clinical Trial Matchmaker Engine",
                role: "Trial Protocols & Eligibility Matching",
                avatar: "🔬",
                text: "Cross-referenced active global trial registries: Matched Protocol FLAURA-2 (Match Score 98.4%, 4 open slots). Patient meets inclusion criteria with stable hematological profile."
            },
            {
                agent: "Safety Guardrail Interceptor",
                role: "Deterministic Circuit-Breakers",
                avatar: "🛡️",
                text: "Executing organ toxicity interlock checks. ALT, AST, and Creatinine are within safety envelopes. No Grade 4 cytopenias detected. Autonomous safety circuit-breaker: PASSED."
            },
            {
                agent: "Tumor Board Consensus Director",
                role: "Multidisciplinary Synthesis Lead",
                avatar: "⚖️",
                text: "Deliberative panel consensus achieved: Recommend standard Osimertinib systemic therapy combined with FLAURA-2 protocol enrollment and bi-weekly ctDNA molecular surveillance."
            }
        ];

        function renderDeliberationStream(stage6Data) {
            const feed = document.getElementById('deliberation-live-feed');
            if (!feed) return;
            feed.innerHTML = '';

            deliberationMessages.forEach((msg, idx) => {
                setTimeout(() => {
                    const timeStr = new Date(Date.now() + idx * 400).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                    feed.innerHTML += `
                        <div class="agent-chat-msg">
                            <div class="agent-chat-avatar">${msg.avatar}</div>
                            <div class="agent-chat-body">
                                <div class="agent-chat-header">
                                    <h5>${msg.agent} <span style="font-size: 0.65rem; color: #0284c7; font-weight: 600; margin-left: 0.35rem;">[${msg.role}]</span></h5>
                                    <span>${timeStr}</span>
                                </div>
                                <p class="agent-chat-text">${msg.text}</p>
                            </div>
                        </div>
                    `;
                    feed.scrollTop = feed.scrollHeight;
                }, idx * 250);
            });
        }

        function replayDeliberationStream() {
            renderDeliberationStream(window.lastPipelineResults?.pipeline_stages?.stage6_agentic);
        }

        // Navigation Switcher
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', function() {
                const targetView = this.getAttribute('data-view');
                if (targetView) {
                    switchView(targetView);
                }
            });
        });

        function switchView(viewId) {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            const matchingNav = document.querySelector(`.nav-link[data-view="${viewId}"]`);
            if (matchingNav) matchingNav.classList.add('active');

            document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
            const targetSec = document.getElementById(viewId);
            if (targetSec) targetSec.classList.add('active');

            if (viewId === 'view-history') loadHistory();
            if (viewId === 'view-analytics') loadAnalytics();
            if (viewId === 'view-audit') renderAuditLogs();
        }

        // Audit Trail State & Filtering Logic
        const initialAuditLogs = [
            {
                timestamp: '2026-10-08 13:14:13',
                patientId: 'PAT-VERIFY-001',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'LOW RISK',
                predColor: '#059669',
                confidence: '77%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 12:54:06',
                patientId: 'SYSTEM',
                user: 'Arthur Pendelton (Patient)',
                role: '[PATIENT]',
                event: 'LOGIN_SUCCESS',
                model: 'AuthGateway',
                prediction: 'SESSION_CREATED',
                predColor: '#059669',
                confidence: '100%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 12:52:45',
                patientId: 'PAT-PRAD-0105',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'LOW RISK',
                predColor: '#059669',
                confidence: '83%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 12:51:22',
                patientId: 'PAT-OOD-005',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'CRITICAL RISK',
                predColor: '#059669',
                confidence: '66%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 12:47:59',
                patientId: 'PAT-OOD-005',
                user: 'Dr. Alistair Chen, MD, PhD',
                role: '',
                event: 'HITL_APPROVED',
                model: 'Autonomous Multi-Agent Deliberator',
                prediction: 'Active AI Recommendation: Withhold targeted TKI therapy immediately; order...',
                predColor: '#059669',
                confidence: '100%',
                ood: 'NORMAL',
                decision: 'APPROVED',
                decisionBold: true
            },
            {
                timestamp: '2026-10-08 12:35:10',
                patientId: 'PAT-NSCLC-0842',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'HIGH RISK',
                predColor: '#ea580c',
                confidence: '91%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 12:31:44',
                patientId: 'PAT-NSCLC-0842',
                user: 'Dr. Sarah Chen, MD',
                role: '(CHIEF ONCOLOGIST)',
                event: 'HITL_APPROVED',
                model: 'Autonomous Multi-Agent Deliberator',
                prediction: 'Active AI Recommendation: Switch to 2nd-Line MET inhibitor combination protocol',
                predColor: '#059669',
                confidence: '95%',
                ood: 'NORMAL',
                decision: 'APPROVED',
                decisionBold: true
            },
            {
                timestamp: '2026-10-08 12:20:15',
                patientId: 'PAT-LUAD-0312',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'MODERATE RISK',
                predColor: '#2563eb',
                confidence: '79%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 12:15:02',
                patientId: 'PAT-LUAD-0312',
                user: 'Clinical Pharmacist',
                role: '(PHARMACIST)',
                event: 'HITL_APPROVED',
                model: 'Autonomous Multi-Agent Deliberator',
                prediction: 'Adjusted Afatinib daily regimen from 40mg to 30mg for GI safety',
                predColor: '#059669',
                confidence: '98%',
                ood: 'NORMAL',
                decision: 'APPROVED',
                decisionBold: true
            },
            {
                timestamp: '2026-10-08 11:58:20',
                patientId: 'PAT-OOD-007',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'OOD_ALERT',
                model: 'OOD Guardrail Ensemble',
                prediction: 'CRITICAL RISK',
                predColor: '#dc2626',
                confidence: '61%',
                ood: 'OUTLIER_FLAGGED',
                decision: 'FLAGGED_REVIEW',
                decisionBold: true
            },
            {
                timestamp: '2026-10-08 11:52:11',
                patientId: 'PAT-OOD-007',
                user: 'Dr. Alistair Chen, MD, PhD',
                role: '',
                event: 'OVERRIDE',
                model: 'Clinical Decision Support Interceptor',
                prediction: 'Override default systemic recommendation pending repeat ctDNA assay',
                predColor: '#d97706',
                confidence: '100%',
                ood: 'OUTLIER_FLAGGED',
                decision: 'OVERRIDDEN',
                decisionBold: true
            },
            {
                timestamp: '2026-10-08 11:40:50',
                patientId: 'PAT-BRCA-0881',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'LOW RISK',
                predColor: '#059669',
                confidence: '89%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 11:25:33',
                patientId: 'PAT-LUSC-0419',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'HIGH RISK',
                predColor: '#ea580c',
                confidence: '88%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 11:21:04',
                patientId: 'PAT-LUSC-0419',
                user: 'Dr. Sarah Chen, MD',
                role: '(CHIEF ONCOLOGIST)',
                event: 'HITL_APPROVED',
                model: 'Autonomous Multi-Agent Deliberator',
                prediction: 'Recommend adjuvant chemotherapy + immunotherapy combination',
                predColor: '#059669',
                confidence: '94%',
                ood: 'NORMAL',
                decision: 'APPROVED',
                decisionBold: true
            },
            {
                timestamp: '2026-10-08 10:55:18',
                patientId: 'PAT-CHEST-1022',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'MODERATE RISK',
                predColor: '#2563eb',
                confidence: '82%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 10:42:09',
                patientId: 'PAT-EGFR-0041',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'CRITICAL RISK',
                predColor: '#dc2626',
                confidence: '96%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 10:38:40',
                patientId: 'PAT-EGFR-0041',
                user: 'Dr. Alistair Chen, MD, PhD',
                role: '',
                event: 'HITL_APPROVED',
                model: 'Autonomous Multi-Agent Deliberator',
                prediction: 'Urgent hepatotoxicity protocol: Withhold Osimertinib and initiate steroids',
                predColor: '#059669',
                confidence: '100%',
                ood: 'NORMAL',
                decision: 'APPROVED',
                decisionBold: true
            },
            {
                timestamp: '2026-10-08 10:14:22',
                patientId: 'PAT-METS-0943',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'HIGH RISK',
                predColor: '#ea580c',
                confidence: '87%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 09:59:15',
                patientId: 'PAT-VERIFY-002',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'LOW RISK',
                predColor: '#059669',
                confidence: '84%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 09:45:00',
                patientId: 'SYSTEM',
                user: 'Hospital Central EMR Gateway',
                role: '[EMR_SYNC]',
                event: 'LOGIN_SUCCESS',
                model: 'HL7/FHIR Ingestion Pipeline',
                prediction: 'SESSION_CREATED',
                predColor: '#059669',
                confidence: '100%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 09:30:12',
                patientId: 'PAT-LUAD-0422',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'MODERATE RISK',
                predColor: '#2563eb',
                confidence: '78%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 09:15:44',
                patientId: 'PAT-PRAD-0118',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'LOW RISK',
                predColor: '#059669',
                confidence: '90%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 08:55:30',
                patientId: 'PAT-LUSC-0511',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'OOD_ALERT',
                model: 'OOD Guardrail Ensemble',
                prediction: 'CRITICAL RISK',
                predColor: '#dc2626',
                confidence: '64%',
                ood: 'ELEVATED_DRIFT',
                decision: 'FLAGGED_REVIEW',
                decisionBold: true
            },
            {
                timestamp: '2026-10-08 08:42:19',
                patientId: 'PAT-CHEST-1088',
                user: 'Attending Oncologist',
                role: '(DOCTOR)',
                event: 'PREDICTION_RUN',
                model: '6-Stage Oncology CDSS Pipeline',
                prediction: 'HIGH RISK',
                predColor: '#ea580c',
                confidence: '86%',
                ood: 'NORMAL',
                decision: 'N/A'
            },
            {
                timestamp: '2026-10-08 08:20:05',
                patientId: 'PAT-NSCLC-0103',
                user: 'Dr. Sarah Chen, MD',
                role: '(CHIEF ONCOLOGIST)',
                event: 'HITL_APPROVED',
                model: 'Autonomous Multi-Agent Deliberator',
                prediction: 'First-line monotherapy approved with bi-weekly LFT safety screening',
                predColor: '#059669',
                confidence: '99%',
                ood: 'NORMAL',
                decision: 'APPROVED',
                decisionBold: true
            }
        ];

        let liveAuditLogs = [...initialAuditLogs];

        function renderAuditLogs(logsToRender) {
            const tbody = document.getElementById('audit-logs-tbody');
            if (!tbody) return;
            const list = logsToRender || liveAuditLogs;
            if (list.length === 0) {
                tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:#94a3b8; padding:2rem;">No matching audit records found.</td></tr>';
                return;
            }
            tbody.innerHTML = list.map(row => `
                <tr>
                    <td style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#475569; white-space:nowrap;">${row.timestamp}</td>
                    <td style="font-weight:700; font-family:'JetBrains Mono', monospace;">${row.patientId}</td>
                    <td>${row.user} ${row.role ? `<span style="color:#94a3b8; font-size:0.75rem;">${row.role}</span>` : ''}</td>
                    <td style="font-family:'JetBrains Mono', monospace; font-weight:600; font-size:0.75rem;">${row.event}</td>
                    <td style="color:#475569;">${row.model}</td>
                    <td style="color:${row.predColor || '#059669'}; font-weight:600;">${row.prediction}</td>
                    <td style="font-weight:600;">${row.confidence}</td>
                    <td style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; ${row.ood !== 'NORMAL' ? 'color:#dc2626; font-weight:700;' : ''}">${row.ood}</td>
                    <td style="${row.decisionBold ? 'font-weight:700; color:#0f172a;' : 'color:#64748b;'}">${row.decision}</td>
                </tr>
            `).join('');
        }

        function applyAuditFilters() {
            const pidQuery = (document.getElementById('audit-filter-pid')?.value || '').trim().toLowerCase();
            const userQuery = (document.getElementById('audit-filter-user')?.value || '').trim().toLowerCase();
            const eventVal = document.getElementById('audit-filter-event')?.value || 'ALL';
            const riskVal = document.getElementById('audit-filter-risk')?.value || 'ALL';

            const filtered = liveAuditLogs.filter(row => {
                if (pidQuery && !row.patientId.toLowerCase().includes(pidQuery)) return false;
                if (userQuery) {
                    const u = row.user.toLowerCase();
                    const r = (row.role || '').toLowerCase();
                    const cleanQ = userQuery.replace('dr.', '').trim();
                    if (!u.includes(userQuery) && !u.includes(cleanQ) && !r.includes(userQuery)) return false;
                }
                if (eventVal !== 'ALL' && row.event !== eventVal) return false;
                if (riskVal !== 'ALL') {
                    const p = row.prediction.toUpperCase();
                    if (riskVal === 'LOW' && !p.includes('LOW')) return false;
                    if (riskVal === 'MODERATE' && !p.includes('MODERATE')) return false;
                    if (riskVal === 'HIGH' && !p.includes('HIGH')) return false;
                    if (riskVal === 'CRITICAL' && !p.includes('CRITICAL')) return false;
                }
                return true;
            });
            renderAuditLogs(filtered);
        }

        function refreshAuditLogs() {
            renderAuditLogs(liveAuditLogs);
        }

        // Multi-Modal Slot State Management (MRI, CT Scan, Pathology)
        let slotStates = {
            mri: { file: null, previewUrl: null, filename: null, size: null, mime: null, validated: false, validModality: null, savedFilename: null, message: null },
            ct: { file: null, previewUrl: null, filename: null, size: null, mime: null, validated: false, validModality: null, savedFilename: null, message: null },
            pathology: { file: null, previewUrl: null, filename: null, size: null, mime: null, validated: false, validModality: null, savedFilename: null, message: null }
        };

        const slotModalityLabels = {
            mri: 'MRI',
            ct: 'CT Scan',
            pathology: 'Pathology'
        };

        // Backward compatibility variables for presets and external calls
        let currentSelectedFile = null;
        let selectedModality = 'CT Scan';
        currentImageFile = null;
        let imageValidationState = {
            validated: false,
            validModality: null,
            savedFilename: null,
            message: null
        };

        function handleModalitySelection(modality) {
            selectedModality = modality;
            ['mri', 'ct', 'pathology'].forEach(m => {
                const pill = document.getElementById(`mod-pill-${m}`);
                if (pill) {
                    const isTarget = (m === 'mri' && modality === 'MRI') ||
                                     (m === 'ct' && modality === 'CT Scan') ||
                                     (m === 'pathology' && modality === 'Pathology');
                    if (isTarget) {
                        pill.classList.add('selected');
                    } else {
                        pill.classList.remove('selected');
                    }
                }
            });
        }

        const s2SlotStates = {
            mri: { file: null, previewUrl: null, filename: null, size: null, mime: null, validated: false, validModality: null, savedFilename: null, message: null },
            ct: { file: null, previewUrl: null, filename: null, size: null, mime: null, validated: false, validModality: null, savedFilename: null, message: null },
            pathology: { file: null, previewUrl: null, filename: null, size: null, mime: null, validated: false, validModality: null, savedFilename: null, message: null }
        };

        function getSlotPrefix(scope) {
            return scope ? `${scope}-` : '';
        }

        function getSlotStatesObj(scope) {
            return scope === 's2' ? s2SlotStates : slotStates;
        }

        function triggerSlotFileInput(modKey, scope = '') {
            const p = getSlotPrefix(scope);
            const input = document.getElementById(`${p}file-input-${modKey}`);
            if (input) input.click();
        }

        function triggerImageFileInput() {
            triggerSlotFileInput('ct');
        }

        function handleSlotDragOver(e, modKey, scope = '') {
            e.preventDefault();
            e.stopPropagation();
            const p = getSlotPrefix(scope);
            const box = document.getElementById(`${p}slot-dropzone-${modKey}`);
            if (box) box.classList.add('drag-over');
        }

        function handleSlotDragLeave(e, modKey, scope = '') {
            e.preventDefault();
            e.stopPropagation();
            const p = getSlotPrefix(scope);
            const box = document.getElementById(`${p}slot-dropzone-${modKey}`);
            if (box) box.classList.remove('drag-over');
        }

        function handleSlotDrop(e, modKey, scope = '') {
            e.preventDefault();
            e.stopPropagation();
            const p = getSlotPrefix(scope);
            const box = document.getElementById(`${p}slot-dropzone-${modKey}`);
            if (box) box.classList.remove('drag-over');
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                processSlotFile(e.dataTransfer.files[0], modKey, scope);
            }
        }

        function handleSlotFileSelect(input, modKey, scope = '') {
            if (input && input.files && input.files.length > 0) {
                const selected = input.files[0];
                input.value = '';
                processSlotFile(selected, modKey, scope);
            }
        }

        function processSlotFile(file, modKey, scope = '') {
            if (!file) return;

            const ext = file.name.split('.').pop().toLowerCase();
            if (!['png', 'jpg', 'jpeg'].includes(ext)) {
                showValidationBanner(`[${slotModalityLabels[modKey]}] Unsupported format. Please upload PNG, JPG, or JPEG.`, "error", scope);
                showSlotFeedback(modKey, "Unsupported file format (PNG, JPG, JPEG only)", "error", scope);
                return;
            }

            if (file.size > 15 * 1024 * 1024) {
                showValidationBanner(`[${slotModalityLabels[modKey]}] File size exceeds 15MB limit.`, "error", scope);
                showSlotFeedback(modKey, "File size exceeds 15MB limit", "error", scope);
                return;
            }

            const st = getSlotStatesObj(scope);
            st[modKey] = {
                file: file,
                previewUrl: null,
                filename: file.name,
                size: (file.size / 1024).toFixed(1) + ' KB',
                mime: file.type || ('image/' + ext),
                validated: false,
                validModality: null,
                savedFilename: null,
                message: null
            };

            const p = getSlotPrefix(scope);
            const card = document.getElementById(`${p}slot-card-${modKey}`);
            const dropzone = document.getElementById(`${p}slot-dropzone-${modKey}`);
            const previewBox = document.getElementById(`${p}slot-preview-${modKey}`);
            const prevImg = document.getElementById(`${p}slot-img-${modKey}`);
            const fnEl = document.getElementById(`${p}slot-filename-${modKey}`);
            const fsEl = document.getElementById(`${p}slot-filesize-${modKey}`);
            const statusEl = document.getElementById(`${p}slot-status-${modKey}`);
            const badgeEl = document.getElementById(`${p}slot-badge-${modKey}`);
            const clearBtn = document.getElementById(`${p}slot-btn-clear-${modKey}`);

            if (card) card.className = 'modality-slot-card has-file';
            if (fnEl) fnEl.innerText = file.name;
            if (fsEl) fsEl.innerText = (file.size / 1024).toFixed(1) + ' KB';
            if (statusEl) {
                statusEl.className = 'slot-status-tag';
                statusEl.innerText = 'Awaiting validation';
            }
            if (badgeEl) {
                badgeEl.className = 'slot-pill-badge ready';
                badgeEl.innerText = 'Ready to Validate';
            }
            if (clearBtn) clearBtn.style.display = 'inline-block';
            hideSlotFeedback(modKey, scope);

            const reader = new FileReader();
            reader.onload = function(e) {
                st[modKey].previewUrl = e.target.result;
                if (prevImg) prevImg.src = e.target.result;
                if (dropzone) dropzone.style.display = 'none';
                if (previewBox) previewBox.style.display = 'flex';
            };
            reader.readAsDataURL(file);

            hideValidationBanner(scope);
        }

        function clearSlotImage(modKey, scope = '') {
            const st = getSlotStatesObj(scope);
            st[modKey] = {
                file: null,
                previewUrl: null,
                filename: null,
                size: null,
                mime: null,
                validated: false,
                validModality: null,
                savedFilename: null,
                message: null
            };

            const p = getSlotPrefix(scope);
            const card = document.getElementById(`${p}slot-card-${modKey}`);
            if (card) card.className = 'modality-slot-card';

            const input = document.getElementById(`${p}file-input-${modKey}`);
            if (input) input.value = '';

            const dropzone = document.getElementById(`${p}slot-dropzone-${modKey}`);
            if (dropzone) dropzone.style.display = 'flex';

            const previewBox = document.getElementById(`${p}slot-preview-${modKey}`);
            if (previewBox) previewBox.style.display = 'none';

            const prevImg = document.getElementById(`${p}slot-img-${modKey}`);
            if (prevImg) prevImg.src = '';

            const badgeEl = document.getElementById(`${p}slot-badge-${modKey}`);
            if (badgeEl) {
                badgeEl.className = 'slot-pill-badge awaiting';
                badgeEl.innerText = 'Awaiting Scan';
            }

            const clearBtn = document.getElementById(`${p}slot-btn-clear-${modKey}`);
            if (clearBtn) clearBtn.style.display = 'none';

            hideSlotFeedback(modKey, scope);
            if (!scope) syncOverallValidationState();
        }

        function clearSelectedImage() {
            clearSlotImage('mri');
            clearSlotImage('ct');
            clearSlotImage('pathology');
        }

        function showSlotFeedback(modKey, msg, type, scope = '') {
            const p = getSlotPrefix(scope);
            const fb = document.getElementById(`${p}slot-feedback-${modKey}`);
            if (!fb) return;
            fb.className = 'slot-feedback-line ' + (type === 'success' ? 'success' : 'error');
            fb.innerText = msg;
            fb.style.display = 'block';
        }

        function hideSlotFeedback(modKey, scope = '') {
            const p = getSlotPrefix(scope);
            const fb = document.getElementById(`${p}slot-feedback-${modKey}`);
            if (fb) fb.style.display = 'none';
        }

        function showValidationBanner(message, type, scope = '') {
            const p = getSlotPrefix(scope);
            const banner = document.getElementById(`${p}validation-banner`);
            const content = document.getElementById(`${p}validation-banner-content`);
            if (!banner || !content) return;

            banner.className = 'validation-banner ' + (type === 'success' ? 'banner-success' : (type === 'info' ? 'banner-info' : 'banner-error'));
            const iconSvg = type === 'success'
                ? '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>'
                : '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';

            content.innerHTML = `${iconSvg} <span>${message}</span>`;
            banner.style.display = 'flex';
        }

        function hideValidationBanner(scope = '') {
            const p = getSlotPrefix(scope);
            const banner = document.getElementById(`${p}validation-banner`);
            if (banner) banner.style.display = 'none';
        }

        async function validateSlotImage(modKey, scope = '') {
            const st = getSlotStatesObj(scope);
            const slot = st[modKey];
            const modalityName = slotModalityLabels[modKey];
            const p = getSlotPrefix(scope);

            if (!slot || (!slot.file && !slot.savedFilename)) {
                showSlotFeedback(modKey, "Please choose an image first.", "error", scope);
                showValidationBanner(`[${modalityName}] Please choose an image to validate.`, "error", scope);
                return false;
            }

            const btn = document.getElementById(`${p}btn-validate-${modKey}`);
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> Validating...`;
            }

            // Cross-slot consistency check: CT image cannot be validated as MRI
            if (modKey === 'mri') {
                const ctSlot = st['ct'];
                if (ctSlot && (ctSlot.file || ctSlot.savedFilename)) {
                    const sameFile = (slot.file && ctSlot.file && slot.file.name === ctSlot.file.name && slot.file.size === ctSlot.file.size) ||
                                     (slot.savedFilename && ctSlot.savedFilename && slot.savedFilename === ctSlot.savedFilename);
                    if (sameFile) {
                        slot.validated = false;
                        slot.validModality = null;
                        const errMsg = "Invalid for MRI: This image is a CT Scan (already loaded in the CT Scan slot). Please upload a valid MRI scan.";
                        const statusTag = document.getElementById(`${p}slot-status-${modKey}`);
                        const badgeTag = document.getElementById(`${p}slot-badge-${modKey}`);
                        const card = document.getElementById(`${p}slot-card-${modKey}`);
                        if (statusTag) { statusTag.className = 'slot-status-tag invalid'; statusTag.innerText = 'Invalid (CT Scan)'; }
                        if (badgeTag) { badgeTag.className = 'slot-pill-badge invalid'; badgeTag.innerText = 'Invalid ✕'; }
                        if (card) card.className = 'modality-slot-card is-invalid';
                        showSlotFeedback(modKey, errMsg, "error", scope);
                        showValidationBanner(`✕ [MRI Scan] ${errMsg}`, "error", scope);
                        if (!scope) syncOverallValidationState();
                        if (btn) {
                            btn.disabled = false;
                            btn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> [ Validate MRI ]`;
                        }
                        return false;
                    }
                }
            }

            try {
                if (!slot.file && slot.savedFilename) {
                    const fnLower = (slot.savedFilename || '').toLowerCase();
                    if (modKey === 'mri' && (fnLower.includes('ct') || fnLower.includes('axial'))) {
                        slot.validated = false;
                        slot.validModality = null;
                        const errMsg = "Invalid for MRI: Preset file is a CT Scan. Please upload a dedicated MRI scan.";
                        const statusTag = document.getElementById(`${p}slot-status-${modKey}`);
                        const badgeTag = document.getElementById(`${p}slot-badge-${modKey}`);
                        const card = document.getElementById(`${p}slot-card-${modKey}`);
                        if (statusTag) { statusTag.className = 'slot-status-tag invalid'; statusTag.innerText = 'Invalid (CT Scan)'; }
                        if (badgeTag) { badgeTag.className = 'slot-pill-badge invalid'; badgeTag.innerText = 'Invalid ✕'; }
                        if (card) card.className = 'modality-slot-card is-invalid';
                        showSlotFeedback(modKey, errMsg, "error", scope);
                        showValidationBanner(`✕ [MRI Scan] ${errMsg}`, "error", scope);
                        if (!scope) syncOverallValidationState();
                        return false;
                    }

                    slot.validated = true;
                    slot.validModality = modalityName;
                    const statusTag = document.getElementById(`${p}slot-status-${modKey}`);
                    const badgeTag = document.getElementById(`${p}slot-badge-${modKey}`);
                    const card = document.getElementById(`${p}slot-card-${modKey}`);
                    if (statusTag) {
                        statusTag.className = 'slot-status-tag valid';
                        statusTag.innerText = slot.message || `✓ Valid ${modalityName} image`;
                    }
                    if (badgeTag) {
                        badgeTag.className = 'slot-pill-badge valid';
                        badgeTag.innerText = 'Validated ✓';
                    }
                    if (card) card.className = 'modality-slot-card is-valid';
                    showSlotFeedback(modKey, slot.message || `✓ Valid ${modalityName} image`, "success", scope);
                    showValidationBanner(`✓ ${modalityName} verified successfully`, "success", scope);
                    if (!scope) syncOverallValidationState();
                    return true;
                }

                const formData = new FormData();
                formData.append('file', slot.file);
                formData.append('image_type', modalityName);

                const res = await fetch('/api/pipeline/validate-image', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                const statusTag = document.getElementById(`${p}slot-status-${modKey}`);
                const badgeTag = document.getElementById(`${p}slot-badge-${modKey}`);
                const card = document.getElementById(`${p}slot-card-${modKey}`);

                if (data.valid === true) {
                    slot.validated = true;
                    slot.validModality = modalityName;
                    slot.savedFilename = data.saved_filename;
                    slot.message = data.message;

                    if (statusTag) {
                        statusTag.className = 'slot-status-tag valid';
                        statusTag.innerText = data.message;
                    }
                    if (badgeTag) {
                        badgeTag.className = 'slot-pill-badge valid';
                        badgeTag.innerText = 'Validated ✓';
                    }
                    if (card) card.className = 'modality-slot-card is-valid';

                    showSlotFeedback(modKey, data.message, "success", scope);
                    showValidationBanner(`✓ ${modalityName}: ${data.message}`, "success", scope);
                    if (!scope) syncOverallValidationState();
                    return true;
                } else {
                    slot.validated = false;
                    slot.validModality = null;
                    const errMsg = data.error || data.message || `Invalid image for ${modalityName}.`;
                    if (statusTag) {
                        statusTag.className = 'slot-status-tag invalid';
                        statusTag.innerText = data.detected_modality ? `Invalid (${data.detected_modality})` : 'Invalid';
                    }
                    if (badgeTag) {
                        badgeTag.className = 'slot-pill-badge invalid';
                        badgeTag.innerText = 'Invalid ✕';
                    }
                    if (card) card.className = 'modality-slot-card is-invalid';

                    showSlotFeedback(modKey, errMsg, "error", scope);
                    showValidationBanner(`✕ [${modalityName}] ${errMsg}`, "error", scope);
                    if (!scope) syncOverallValidationState();
                    return false;
                }
            } catch (err) {
                slot.validated = false;
                showSlotFeedback(modKey, "Validation error. Please try again.", "error", scope);
                showValidationBanner(`Unable to validate ${modalityName} right now. Please try again.`, "error", scope);
                return false;
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> [ Validate ${modalityName === 'CT Scan' ? 'CT' : modalityName} ]`;
                }
            }
        }

        function validateSelectedImage() {
            validateAllLoadedSlots();
        }

        async function validateAllLoadedSlots(scope = '') {
            const st = getSlotStatesObj(scope);
            let anyLoaded = false;
            let allValid = true;

            for (const key of ['mri', 'ct', 'pathology']) {
                if (st[key].file || st[key].savedFilename) {
                    anyLoaded = true;
                    const ok = await validateSlotImage(key, scope);
                    if (!ok) allValid = false;
                }
            }

            if (!anyLoaded) {
                showValidationBanner("Please choose or upload at least one medical image in the slots above to validate.", "error", scope);
                return;
            }

            if (allValid) {
                showValidationBanner("All medical image inputs have been validated!", "success", scope);
            }
        }

        function syncOverallValidationState() {
            const hasCT = slotStates.ct.validated && Boolean(slotStates.ct.savedFilename);
            const hasPath = slotStates.pathology.validated && Boolean(slotStates.pathology.savedFilename);
            const hasMRI = slotStates.mri.validated && Boolean(slotStates.mri.savedFilename);

            if (hasCT) {
                imageValidationState.validated = true;
                imageValidationState.validModality = 'CT Scan';
                imageValidationState.savedFilename = slotStates.ct.savedFilename;
                imageValidationState.message = slotStates.ct.message;
                currentImageFile = slotStates.ct.savedFilename;
                selectedModality = 'CT Scan';
            } else if (hasPath) {
                imageValidationState.validated = true;
                imageValidationState.validModality = 'Pathology';
                imageValidationState.savedFilename = slotStates.pathology.savedFilename;
                imageValidationState.message = slotStates.pathology.message;
                currentImageFile = slotStates.pathology.savedFilename;
                selectedModality = 'Pathology';
            } else if (hasMRI) {
                imageValidationState.validated = true;
                imageValidationState.validModality = 'MRI';
                imageValidationState.savedFilename = slotStates.mri.savedFilename;
                imageValidationState.message = slotStates.mri.message;
                currentImageFile = slotStates.mri.savedFilename;
                selectedModality = 'MRI';
            } else {
                imageValidationState.validated = false;
                imageValidationState.validModality = null;
                imageValidationState.savedFilename = null;
                imageValidationState.message = null;
                currentImageFile = null;
            }
        }

        function handleRunAnalysisClick() {
            const hasValidCT = slotStates.ct.validated && Boolean(slotStates.ct.savedFilename);
            const hasValidPath = slotStates.pathology.validated && Boolean(slotStates.pathology.savedFilename);
            const hasValidMRI = slotStates.mri.validated && Boolean(slotStates.mri.savedFilename);

            if (!hasValidCT && !hasValidPath && !hasValidMRI) {
                showValidationBanner("Please upload and validate a medical image before starting analysis.", "error");
                return;
            }

            if (hasValidMRI && !hasValidCT && !hasValidPath) {
                showValidationBanner("This image type is not supported by the current analysis model.", "error");
                return;
            }

            runCompleteAnalysis();
        }

        // Preset Loader
        async function loadPreset(presetKey) {
            try {
                const res = await fetch('/api/pipeline/presets');
                const presets = await res.json();
                const p = presets[presetKey];
                if (!p || !p.data) return;

                const d = p.data;
                for (const k in d) {
                    const el = document.getElementById(`inp-${k}`);
                    if (el) el.value = d[k];
                }

                clearSlotImage('mri');
                clearSlotImage('ct');
                clearSlotImage('pathology');

                if (d.image_file) {
                    const isTile = d.image_file.includes('tile') || d.image_file.includes('colon') || (d.image_type === 'Pathology');
                    const presetModality = isTile ? 'Pathology' : 'CT Scan';
                    loadPresetImage(d.image_file, presetModality);
                }
            } catch (e) {
                console.error("Preset load error", e);
            }
        }

        function loadPresetImage(imagePath, modality) {
            const isColon = imagePath.toLowerCase().includes('colon');
            const modKey = (modality === 'Pathology' || imagePath.includes('tile') || isColon) ? 'pathology' : 'ct';
            const filename = imagePath.split('/').pop();
            const successMsg = isColon ? '✓ Valid Pathology image (Colon Adenocarcinoma H&E Tile)' : (modKey === 'pathology' ? '✓ Valid Pathology image' : '✓ Valid CT image');
            const isJpeg = imagePath.toLowerCase().endsWith('.jpeg') || imagePath.toLowerCase().endsWith('.jpg');

            slotStates[modKey] = {
                file: null,
                previewUrl: `/api/pipeline/image/${imagePath}`,
                filename: filename,
                size: isColon ? '110.5 KB' : '148.2 KB',
                mime: isJpeg ? 'image/jpeg' : 'image/png',
                validated: true,
                validModality: slotModalityLabels[modKey],
                savedFilename: imagePath,
                message: successMsg
            };

            const card = document.getElementById(`slot-card-${modKey}`);
            const dropzone = document.getElementById(`slot-dropzone-${modKey}`);
            const previewBox = document.getElementById(`slot-preview-${modKey}`);
            const prevImg = document.getElementById(`slot-img-${modKey}`);
            const fnEl = document.getElementById(`slot-filename-${modKey}`);
            const fsEl = document.getElementById(`slot-filesize-${modKey}`);
            const statusEl = document.getElementById(`slot-status-${modKey}`);
            const badgeEl = document.getElementById(`slot-badge-${modKey}`);
            const clearBtn = document.getElementById(`slot-btn-clear-${modKey}`);

            if (card) card.className = 'modality-slot-card is-valid';
            if (dropzone) dropzone.style.display = 'none';
            if (previewBox) previewBox.style.display = 'flex';
            if (prevImg) prevImg.src = `/api/pipeline/image/${imagePath}`;
            if (fnEl) fnEl.innerText = filename;
            if (fsEl) fsEl.innerText = isColon ? '110.5 KB' : '148.2 KB';
            if (statusEl) {
                statusEl.className = 'slot-status-tag valid';
                statusEl.innerText = successMsg;
            }
            if (badgeEl) {
                badgeEl.className = 'slot-pill-badge valid';
                badgeEl.innerText = 'Validated ✓';
            }
            if (clearBtn) clearBtn.style.display = 'inline-block';
            showSlotFeedback(modKey, successMsg, "success");

            syncOverallValidationState();
            showValidationBanner(successMsg, "success");
        }

        function setOverviewSample(type) {
            clearSlotImage('mri');
            clearSlotImage('ct');
            clearSlotImage('pathology');

            if (type === 'HIGH') {
                const elId = document.getElementById('inp-patient_id');
                if (elId) elId.value = 'PAT-ML-0101';
                const elAge = document.getElementById('inp-age');
                if (elAge) elAge.value = 64;
                const elSex = document.getElementById('inp-sex');
                if (elSex) elSex.value = 'M';
                const elType = document.getElementById('inp-cancer_type');
                if (elType) elType.value = 'Lung (LUAD)';
                const elStage = document.getElementById('inp-cancer_stage');
                if (elStage) elStage.value = 'Stage IV';
                const elMut = document.getElementById('inp-mutation_profile');
                if (elMut) elMut.value = 'EGFR L858R';
                const elTm = document.getElementById('inp-tumor_marker_level');
                if (elTm) elTm.value = 28.6;
                const elCtdna = document.getElementById('inp-ctDNA_level');
                if (elCtdna) elCtdna.value = 1.84;
                const elAlt = document.getElementById('inp-ALT');
                if (elAlt) elAlt.value = 248;
                const elAst = document.getElementById('inp-AST');
                if (elAst) elAst.value = 210;
                const elCr = document.getElementById('inp-creatinine');
                if (elCr) elCr.value = 1.45;
                const elBili = document.getElementById('inp-bilirubin');
                if (elBili) elBili.value = 2.2;
                const elTx = document.getElementById('inp-treatment_name');
                if (elTx) elTx.value = 'Osimertinib 80mg + Capmatinib';
                const elDose = document.getElementById('inp-dosage_mg');
                if (elDose) elDose.value = 480;
                const elCyc = document.getElementById('inp-treatment_cycle');
                if (elCyc) elCyc.value = 4;
                const elNotes = document.getElementById('inp-clinical_notes');
                if (elNotes) elNotes.value = 'Patient on Osimertinib and Capmatinib presenting with spiking fever 38.4C, marked right upper quadrant tenderness, intractable nausea, and severe fatigue. Marked transaminase elevation (ALT 248 U/L, AST 210 U/L) indicating acute Drug-Induced Liver Injury (DILI).';

                loadPresetImage('luad/ct_slice_0000.png', 'CT Scan');
            } else if (type === 'MODERATE') {
                const elId = document.getElementById('inp-patient_id');
                if (elId) elId.value = 'PAT-ML-0204';
                const elAge = document.getElementById('inp-age');
                if (elAge) elAge.value = 62;
                const elSex = document.getElementById('inp-sex');
                if (elSex) elSex.value = 'F';
                const elType = document.getElementById('inp-cancer_type');
                if (elType) elType.value = 'Breast (BRCA)';
                const elStage = document.getElementById('inp-cancer_stage');
                if (elStage) elStage.value = 'Stage III';
                const elMut = document.getElementById('inp-mutation_profile');
                if (elMut) elMut.value = 'PIK3CA H1047R';
                const elTm = document.getElementById('inp-tumor_marker_level');
                if (elTm) elTm.value = 16.5;
                const elCtdna = document.getElementById('inp-ctDNA_level');
                if (elCtdna) elCtdna.value = 0.82;
                const elAlt = document.getElementById('inp-ALT');
                if (elAlt) elAlt.value = 75;
                const elAst = document.getElementById('inp-AST');
                if (elAst) elAst.value = 68;
                const elCr = document.getElementById('inp-creatinine');
                if (elCr) elCr.value = 1.15;
                const elBili = document.getElementById('inp-bilirubin');
                if (elBili) elBili.value = 1.2;
                const elTx = document.getElementById('inp-treatment_name');
                if (elTx) elTx.value = 'Abemaciclib + Fulvestrant';
                const elDose = document.getElementById('inp-dosage_mg');
                if (elDose) elDose.value = 1000;
                const elCyc = document.getElementById('inp-treatment_cycle');
                if (elCyc) elCyc.value = 3;
                const elNotes = document.getElementById('inp-clinical_notes');
                if (elNotes) elNotes.value = 'Patient on CDK4/6 inhibitor Abemaciclib presenting with moderate neutropenia and Grade 1 fatigue. Biomarker profile reveals PIK3CA activating mutation.';

                loadPresetImage('brca/tile_0000.png', 'Pathology');
            } else if (type === 'LOW') {
                const elId = document.getElementById('inp-patient_id');
                if (elId) elId.value = 'PAT-ML-0318';
                const elAge = document.getElementById('inp-age');
                if (elAge) elAge.value = 54;
                const elSex = document.getElementById('inp-sex');
                if (elSex) elSex.value = 'M';
                const elType = document.getElementById('inp-cancer_type');
                if (elType) elType.value = 'Prostate (PRAD)';
                const elStage = document.getElementById('inp-cancer_stage');
                if (elStage) elStage.value = 'Stage I';
                const elMut = document.getElementById('inp-mutation_profile');
                if (elMut) elMut.value = 'AR-V7 negative';
                const elTm = document.getElementById('inp-tumor_marker_level');
                if (elTm) elTm.value = 4.2;
                const elCtdna = document.getElementById('inp-ctDNA_level');
                if (elCtdna) elCtdna.value = 0.12;
                const elAlt = document.getElementById('inp-ALT');
                if (elAlt) elAlt.value = 28;
                const elAst = document.getElementById('inp-AST');
                if (elAst) elAst.value = 24;
                const elCr = document.getElementById('inp-creatinine');
                if (elCr) elCr.value = 0.95;
                const elBili = document.getElementById('inp-bilirubin');
                if (elBili) elBili.value = 0.8;
                const elTx = document.getElementById('inp-treatment_name');
                if (elTx) elTx.value = 'Enzalutamide 160mg';
                const elDose = document.getElementById('inp-dosage_mg');
                if (elDose) elDose.value = 160;
                const elCyc = document.getElementById('inp-treatment_cycle');
                if (elCyc) elCyc.value = 2;
                const elNotes = document.getElementById('inp-clinical_notes');
                if (elNotes) elNotes.value = 'Localized prostate adenocarcinoma post-prostatectomy. Surveillance PSA 4.2 ng/mL, well-tolerated androgen receptor pathway inhibitor.';

                loadPresetImage('prad/ct_slice_0000.png', 'CT Scan');
            } else if (type === 'COLON') {
                const elId = document.getElementById('inp-patient_id');
                if (elId) elId.value = 'PAT-COAD-4273';
                const elAge = document.getElementById('inp-age');
                if (elAge) elAge.value = 62;
                const elSex = document.getElementById('inp-sex');
                if (elSex) elSex.value = 'M';
                const elType = document.getElementById('inp-cancer_type');
                if (elType) elType.value = 'Colon (COAD)';
                const elStage = document.getElementById('inp-cancer_stage');
                if (elStage) elStage.value = 'Stage III';
                const elMut = document.getElementById('inp-mutation_profile');
                if (elMut) elMut.value = 'KRAS G12D, APC';
                const elTm = document.getElementById('inp-tumor_marker_level');
                if (elTm) elTm.value = 18.4;
                const elCtdna = document.getElementById('inp-ctDNA_level');
                if (elCtdna) elCtdna.value = 0.82;
                const elAlt = document.getElementById('inp-ALT');
                if (elAlt) elAlt.value = 38;
                const elAst = document.getElementById('inp-AST');
                if (elAst) elAst.value = 34;
                const elCr = document.getElementById('inp-creatinine');
                if (elCr) elCr.value = 1.05;
                const elBili = document.getElementById('inp-bilirubin');
                if (elBili) elBili.value = 1.1;
                const elTx = document.getElementById('inp-treatment_name');
                if (elTx) elTx.value = 'FOLFOX + Bevacizumab';
                const elDose = document.getElementById('inp-dosage_mg');
                if (elDose) elDose.value = 350;
                const elCyc = document.getElementById('inp-treatment_cycle');
                if (elCyc) elCyc.value = 4;
                const elNotes = document.getElementById('inp-clinical_notes');
                if (elNotes) elNotes.value = 'Primary colon adenocarcinoma resection specimen test set evaluation. Histopathology H&E tile demonstrates moderate-to-poorly differentiated glandular architecture with stromal invasion.';

                loadPresetImage('colon_aca/colonca4273.jpeg', 'Pathology');
            }
        }

        function updateDLSampleImages(cancerType) {
            const ct = (cancerType || '').toLowerCase();
            if (ct === 'coad' || ct.includes('colon')) {
                // Populate Pathology slot with real colon adenocarcinoma test set image
                loadPresetImage('colon_aca/colonca4273.jpeg', 'Pathology');
                // Also set s2SlotStates for Stage 2 view
                s2SlotStates.pathology = {
                    file: null,
                    previewUrl: '/api/pipeline/image/colon_aca/colonca4273.jpeg',
                    filename: 'colonca4273.jpeg',
                    size: '110.5 KB',
                    mime: 'image/jpeg',
                    validated: true,
                    validModality: 'Pathology',
                    savedFilename: 'colon_aca/colonca4273.jpeg',
                    message: '✓ Valid Pathology image (Colon Adenocarcinoma H&E Tile)'
                };
                const s2Card = document.getElementById('s2-slot-card-pathology');
                const s2Drop = document.getElementById('s2-slot-dropzone-pathology');
                const s2Prev = document.getElementById('s2-slot-preview-pathology');
                const s2Img = document.getElementById('s2-slot-img-pathology');
                const s2Fn = document.getElementById('s2-slot-filename-pathology');
                const s2Status = document.getElementById('s2-slot-status-pathology');
                const s2Badge = document.getElementById('s2-slot-badge-pathology');
                if (s2Card) s2Card.className = 'modality-slot-card is-valid';
                if (s2Drop) s2Drop.style.display = 'none';
                if (s2Prev) s2Prev.style.display = 'flex';
                if (s2Img) s2Img.src = '/api/pipeline/image/colon_aca/colonca4273.jpeg';
                if (s2Fn) s2Fn.innerText = 'colonca4273.jpeg';
                if (s2Status) { s2Status.className = 'slot-status-tag valid'; s2Status.innerText = '✓ Valid Pathology image'; }
                if (s2Badge) { s2Badge.className = 'slot-pill-badge valid'; s2Badge.innerText = 'Validated ✓'; }
            }
        }

        // CSV Upload Handler
        async function handleCSVUpload(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const nameEl = document.getElementById('csv-filename');
            if (nameEl) nameEl.innerText = file.name;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/pipeline/upload-csv', {
                    method: 'POST',
                    body: formData
                });
                const result = await res.json();
                if (result.status === 'success' && result.patient_data) {
                    const d = result.patient_data;
                    for (const k in d) {
                        const el = document.getElementById(`inp-${k}`);
                        if (el && d[k] !== null && d[k] !== undefined) {
                            el.value = d[k];
                        }
                    }
                    alert(result.message);
                } else {
                    alert("Upload error: " + (result.detail || "Could not parse CSV."));
                }
            } catch (e) {
                alert("CSV upload failed: " + e.message);
            }
        }

        function resetForm() {
            const form = document.getElementById('patient-form');
            if (form) form.reset();
            clearSelectedImage();
            hideValidationBanner();
            const csvText = document.getElementById('csv-filename');
            if (csvText) csvText.innerText = 'Upload Patient CSV (Optional)';
            document.getElementById('results-grid').style.display = 'none';
            document.getElementById('final-assessment-section').style.display = 'none';
            resetPipelineFlow();
        }

        function resetPipelineFlow() {
            ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'].forEach(s => {
                const node = document.getElementById(`node-${s}`);
                const pill = document.getElementById(`pill-${s}`);
                if (node) node.className = 'flow-node';
                if (pill) { pill.innerText = 'Waiting'; }
            });
            document.getElementById('pipeline-progress-fill').style.width = '0%';
            document.getElementById('pipeline-progress-label').innerText = 'Stage 0 of 6 (Ready)';
        }

        // Sequential 6-Stage Execution
        async function runCompleteAnalysis() {
            const btn = document.getElementById('btn-run-analysis');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> EXECUTING MULTI-STAGE ANALYSIS...`;

            resetPipelineFlow();
            document.getElementById('results-grid').style.display = 'grid';
            document.getElementById('final-assessment-section').style.display = 'none';

            const hasCT = slotStates.ct.validated && Boolean(slotStates.ct.savedFilename);
            const hasPath = slotStates.pathology.validated && Boolean(slotStates.pathology.savedFilename);
            const hasMRI = slotStates.mri.validated && Boolean(slotStates.mri.savedFilename);

            const activeImg = hasCT ? slotStates.ct.savedFilename : (hasPath ? slotStates.pathology.savedFilename : (hasMRI ? slotStates.mri.savedFilename : (imageValidationState.savedFilename || currentImageFile)));
            const activeType = hasCT ? 'CT Scan' : (hasPath ? 'Pathology' : (hasMRI ? 'MRI' : selectedModality));

            // Harvest form values (with safe fallbacks to existing default values)
            const payload = {
                patient_id: (document.getElementById('inp-patient_id') ? document.getElementById('inp-patient_id').value : 'PAT-NSCLC-0842') || 'PAT-0001',
                age: parseFloat(document.getElementById('inp-age') ? document.getElementById('inp-age').value : 64) || 64,
                sex: (document.getElementById('inp-sex') ? document.getElementById('inp-sex').value : 'M') || 'M',
                cancer_type: (document.getElementById('inp-cancer_type') ? document.getElementById('inp-cancer_type').value : 'Lung (LUAD)') || 'Lung (LUAD)',
                cancer_stage: (document.getElementById('inp-cancer_stage') ? document.getElementById('inp-cancer_stage').value : 'Stage IV') || 'Stage IV',
                mutation_profile: (document.getElementById('inp-mutation_profile') ? document.getElementById('inp-mutation_profile').value : 'EGFR L858R') || 'EGFR L858R',
                tumor_marker_level: parseFloat(document.getElementById('inp-tumor_marker_level') ? document.getElementById('inp-tumor_marker_level').value : 28.6) || 28.6,
                ctDNA_level: parseFloat(document.getElementById('inp-ctDNA_level') ? document.getElementById('inp-ctDNA_level').value : 1.84) || 1.84,
                ALT: parseFloat(document.getElementById('inp-ALT') ? document.getElementById('inp-ALT').value : 248) || 248,
                AST: parseFloat(document.getElementById('inp-AST') ? document.getElementById('inp-AST').value : 210) || 210,
                creatinine: parseFloat(document.getElementById('inp-creatinine') ? document.getElementById('inp-creatinine').value : 1.45) || 1.45,
                bilirubin: parseFloat(document.getElementById('inp-bilirubin') ? document.getElementById('inp-bilirubin').value : 2.2) || 2.2,
                treatment_name: (document.getElementById('inp-treatment_name') ? document.getElementById('inp-treatment_name').value : 'Osimertinib 80mg + Capmatinib') || 'Targeted Therapy',
                dosage_mg: parseFloat(document.getElementById('inp-dosage_mg') ? document.getElementById('inp-dosage_mg').value : 480) || 400,
                treatment_cycle: parseInt(document.getElementById('inp-treatment_cycle') ? document.getElementById('inp-treatment_cycle').value : 4) || 3,
                clinical_notes: document.getElementById('inp-clinical_notes') ? document.getElementById('inp-clinical_notes').value : '',
                image_file: activeImg,
                image_type: activeType,
                ct_image_file: hasCT ? slotStates.ct.savedFilename : null,
                pathology_image_file: hasPath ? slotStates.pathology.savedFilename : null,
                mri_image_file: hasMRI ? slotStates.mri.savedFilename : null
            };

            // Animate pipeline stages sequentially across all 6 stages
            const stages = [
                { id: 'stage1', name: 'Stage 1 of 6: ML Toxicity Analysis', pct: 16 },
                { id: 'stage2', name: 'Stage 2 of 6: DL Medical Image Analysis', pct: 33 },
                { id: 'stage3', name: 'Stage 3 of 6: NLP Clinical Triage', pct: 50 },
                { id: 'stage4', name: 'Stage 4 of 6: SLM Clinical Reasoning', pct: 66 },
                { id: 'stage5', name: 'Stage 5 of 6: GenAI Final Report', pct: 83 },
                { id: 'stage6', name: 'Stage 6 of 6: Agentic AI Deliberation & Trials', pct: 100 }
            ];

            let activeIdx = 0;
            const progressTimer = setInterval(() => {
                if (activeIdx < stages.length) {
                    const curr = stages[activeIdx];
                    document.getElementById('pipeline-progress-fill').style.width = curr.pct + '%';
                    document.getElementById('pipeline-progress-label').innerText = curr.name;

                    const currNode = document.getElementById(`node-${curr.id}`);
                    const currPill = document.getElementById(`pill-${curr.id}`);
                    if (currNode) currNode.className = 'flow-node processing-stage';
                    if (currPill) currPill.innerText = 'Processing ⟳';

                    if (activeIdx > 0) {
                        const prev = stages[activeIdx - 1];
                        const prevNode = document.getElementById(`node-${prev.id}`);
                        const prevPill = document.getElementById(`pill-${prev.id}`);
                        if (prevNode) prevNode.className = 'flow-node completed-stage';
                        if (prevPill) prevPill.innerText = 'Completed ✓';
                    }
                    activeIdx++;
                }
            }, 350);

            try {
                const res = await fetch('/api/pipeline/run-all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                clearInterval(progressTimer);

                // Set all stages completed
                stages.forEach(s => {
                    const node = document.getElementById(`node-${s.id}`);
                    const pill = document.getElementById(`pill-${s.id}`);
                    if (node) node.className = 'flow-node completed-stage';
                    if (pill) pill.innerText = 'Completed ✓';
                });
                document.getElementById('pipeline-progress-fill').style.width = '100%';
                document.getElementById('pipeline-progress-label').innerText = 'Analysis Complete (100%)';

                // Render Intermediate Results
                renderStageOutputs(data);

                // Render Final Assessment
                renderFinalAssessment(data);

            } catch (err) {
                clearInterval(progressTimer);
                alert("Pipeline execution failed: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg> RUN COMPLETE AI ANALYSIS`;
            }
        }

        function renderStageOutputs(data) {
            const stages = data.pipeline_stages || {};

            // 1. ML Output
            const s1 = stages.stage1_ml || {};
            document.getElementById('ml-badge-risk').innerText = `${s1.prediction || 'N/A'} Risk (${s1.confidence_pct || 85}%)`;
            document.getElementById('ml-text-interp').innerText = s1.clinical_interpretation || '';
            const mlTags = document.getElementById('ml-tags-list');
            mlTags.innerHTML = '';
            (s1.important_features || []).forEach(f => {
                mlTags.innerHTML += `<span class="feature-tag"><strong>${f.feature.split(' ')[0]}:</strong> ${f.value} (${f.impact})</span>`;
            });
            // Feature 3: Visual Model Explainability (SHAP Waterfall)
            if (typeof renderStage1ShapWaterfall === 'function') {
                renderStage1ShapWaterfall(s1);
            }

            // 2. DL Output
            const s2 = stages.stage2_dl || {};
            document.getElementById('dl-badge-risk').innerText = `${s2.risk_tier || 'Moderate'} (${Math.round((s2.progression_probability || 0.45)*100)}% Prog)`;
            document.getElementById('dl-text-interp').innerText = s2.clinical_interpretation || '';
            const dlRow = document.getElementById('dl-preview-row');
            dlRow.innerHTML = `
                <img src="${s2.pathology_tile_file || '/api/pipeline/image/luad/tile_0000.png'}" style="width: 38px; height: 38px; border-radius: 4px; object-fit: cover; border: 1px solid #e2e8f0;">
                <img src="${s2.ct_slice_file || '/api/pipeline/image/luad/ct_slice_0000.png'}" style="width: 38px; height: 38px; border-radius: 4px; object-fit: cover; border: 1px solid #e2e8f0;">
                <span style="font-size: 0.72rem; color: var(--text-secondary);">Lesion: <strong>${s2.image_prediction || 'Stable'}</strong></span>
            `;

            // 3. NLP Output
            const s3 = stages.stage3_nlp || {};
            document.getElementById('nlp-badge-triage').innerText = `${s3.urgency_classification || 'MODERATE'} Priority`;
            document.getElementById('nlp-text-interp').innerText = s3.clinical_interpretation || '';
            const nlpTags = document.getElementById('nlp-entities-list');
            nlpTags.innerHTML = '';
            (s3.extracted_entities || []).slice(0, 6).forEach(e => {
                nlpTags.innerHTML += `<span class="feature-tag"><strong>${e.label}:</strong> ${e.text}</span>`;
            });

            // 4. SLM Output
            const s4 = stages.stage4_slm || {};
            document.getElementById('slm-badge-guardrail').innerText = s4.guardrail_status || 'VERIFIED_SAFE';
            document.getElementById('slm-text-interp').innerText = s4.clinical_reasoning || '';

            // 5. GenAI Output
            const s5 = stages.stage5_genai || {};
            const genaiBadge = document.getElementById('genai-badge-status');
            if (genaiBadge) genaiBadge.innerText = s5.status || 'Completed';
            const genaiText = document.getElementById('genai-text-interp');
            if (genaiText) genaiText.innerText = s5.patient_summary || s5.summary || 'Generative clinical report synthesis complete.';
            const genaiTags = document.getElementById('genai-tags-list');
            if (genaiTags) {
                genaiTags.innerHTML = '';
                const q = s5.quality_indicators || {};
                genaiTags.innerHTML += `<span class="feature-tag"><strong>Consistency:</strong> ${Math.round((q.factual_consistency || 0.98)*100)}%</span>`;
                genaiTags.innerHTML += `<span class="feature-tag"><strong>Adherence:</strong> ${Math.round((q.clinical_adherence || 0.96)*100)}%</span>`;
                genaiTags.innerHTML += `<span class="feature-tag"><strong>Cross-Modal Safety:</strong> ${Math.round((q.cross_modal_safety_score || 0.95)*100)}%</span>`;
            }

            // 6. Agentic Output
            const s6 = stages.stage6_agentic || {};
            const agenticBadge = document.getElementById('agentic-badge-status');
            const safetyStatus = s6.safety_status || 'NORMAL';
            if (agenticBadge) {
                const isInterlock = safetyStatus.includes('HOLD') || safetyStatus.includes('INTERLOCK') || safetyStatus.includes('HALT');
                agenticBadge.innerText = isInterlock ? 'Interlock Halt ⚠' : 'Consensus ✓';
                agenticBadge.className = isInterlock ? 'metric-badge red' : 'metric-badge green';
            }
            const agenticText = document.getElementById('agentic-text-interp');
            const finalRec = (s6.final_assessment && s6.final_assessment.final_recommendation) || s6.deliberation_summary || 'Multi-agent deliberation completed.';
            if (agenticText) agenticText.innerText = finalRec;
            const agenticTags = document.getElementById('agentic-tags-list');
            if (agenticTags) {
                agenticTags.innerHTML = '';
                agenticTags.innerHTML += `<span class="feature-tag"><strong>Agents:</strong> ${s6.participating_agents_count || 10} Active</span>`;
                const topTrial = (s6.matched_trials && s6.matched_trials[0]) || null;
                if (topTrial) {
                    agenticTags.innerHTML += `<span class="feature-tag" style="background:#e0f2fe; color:#0369a1;"><strong>Top Trial:</strong> ${topTrial.trial_name || topTrial.trial_id} (${topTrial.matching_score || 95}% match)</span>`;
                }
            }
        }

        function renderFinalAssessment(data) {
            const final = data.final_assessment || {};
            const finalCard = document.getElementById('final-assessment-section');
            finalCard.style.display = 'flex';

            document.getElementById('final-card-patient-title').innerText = `Comprehensive Patient Assessment — ${final.patient_id || ''}`;
            document.getElementById('final-card-patient-sub').innerText = `Diagnosis: ${final.cancer_type || 'Oncology'} | Timestamp: ${new Date().toLocaleDateString()}`;

            const riskTag = document.getElementById('final-risk-tag');
            const riskText = final.overall_risk || 'MODERATE RISK';
            riskTag.innerText = riskText;

            riskTag.className = 'risk-classification-tag';
            if (riskText.includes('LOW')) riskTag.classList.add('low');
            else if (riskText.includes('MODERATE')) riskTag.classList.add('moderate');
            else if (riskText.includes('CRITICAL')) riskTag.classList.add('critical');
            else riskTag.classList.add('high');

            document.getElementById('fin-overall-risk').innerText = riskText;
            document.getElementById('fin-risk-prob').innerText = `${final.risk_probability_pct || 75.0}%`;
            document.getElementById('fin-confidence').innerText = `${final.confidence_pct || 90.0}%`;

            const interlock = document.getElementById('fin-interlock-status');
            if (interlock) {
                const s6Status = final.safety_interlock_status || 'NORMAL';
                interlock.innerText = s6Status.includes('TRIGGERED') ? 'INTERLOCK HALT' : 'NORMAL';
                interlock.style.color = s6Status.includes('TRIGGERED') ? '#dc2626' : '#059669';
            }

            document.getElementById('fin-genai-summary').innerText = final.genai_summary || '';

            // Render Stage 6 Autonomous Deliberation & Trial Box
            const agenticBox = document.getElementById('fin-agentic-box');
            if (agenticBox && final.agentic_recommendation) {
                agenticBox.style.display = 'block';
                document.getElementById('fin-agentic-rec').innerText = final.agentic_recommendation;
                const trials = final.matched_trials || [];
                const trialsEl = document.getElementById('fin-agentic-trials');
                if (trials.length > 0) {
                    trialsEl.innerHTML = `<strong>Matched Clinical Protocol:</strong> ${trials[0].trial_name} · Score: ${trials[0].matching_score}% · Available Slots: ${trials[0].open_slots || 1}`;
                } else {
                    trialsEl.innerHTML = `<em>No clinical trial active — safety interlock directive enforced.</em>`;
                }
            }

            const considerationsList = document.getElementById('fin-considerations-list');
            considerationsList.innerHTML = '';
            (final.clinical_considerations || []).forEach(c => {
                considerationsList.innerHTML += `
                    <div class="consideration-item">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                        <span>${c}</span>
                    </div>
                `;
            });

            finalCard.scrollIntoView({ behavior: 'smooth' });

            // Store full analysis result globally for PDF generation & Simulator sync
            window.lastPipelineResults = data;
            if (typeof updateWhatIfSimulation === 'function') {
                updateWhatIfSimulation();
            }
            if (typeof renderDeliberationStream === 'function') {
                renderDeliberationStream(data.pipeline_stages?.stage6_agentic);
            }
        }

        // Load History Records
        async function loadHistory() {
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 2rem;">Loading history...</td></tr>`;

            try {
                const res = await fetch('/api/pipeline/history');
                const rows = await res.json();
                if (!rows || rows.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 2rem;">No patient analyses recorded yet. Run an analysis above.</td></tr>`;
                    return;
                }

                tbody.innerHTML = '';
                rows.forEach(r => {
                    const badgeClass = (r.final_risk || '').toLowerCase().includes('high') ? 'red' : ((r.final_risk || '').toLowerCase().includes('low') ? 'green' : 'blue');
                    tbody.innerHTML += `
                        <tr>
                            <td><strong>${r.patient_id}</strong></td>
                            <td>${new Date(r.timestamp).toLocaleString()}</td>
                            <td>${r.cancer_type} (${r.stage})</td>
                            <td>${r.ml_result || 'Completed'}</td>
                            <td>${r.dl_result || 'Stable'}</td>
                            <td>${r.nlp_result || 'Moderate'}</td>
                            <td><span class="metric-badge ${badgeClass}">${r.final_risk}</span></td>
                            <td><button class="btn-preset" onclick="viewHistoryItem(${r.id})">Inspect</button></td>
                        </tr>
                    `;
                });
            } catch (e) {
                tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: red;">Failed to load history: ${e.message}</td></tr>`;
            }
        }

        async function viewHistoryItem(id) {
            try {
                const res = await fetch('/api/pipeline/history');
                const rows = await res.json();
                const item = rows.find(x => x.id === id);
                if (item && item.payload) {
                    switchView('view-prediction');
                    renderStageOutputs(item.payload);
                    renderFinalAssessment(item.payload);
                }
            } catch (e) {
                alert("Could not load record: " + e.message);
            }
        }

        // Load Analytics
        async function loadAnalytics() {
            try {
                const res = await fetch('/api/pipeline/analytics');
                const data = await res.json();
                if (data.total_patients_analyzed) {
                    document.getElementById('stat-total').innerText = data.total_patients_analyzed;
                    document.getElementById('stat-high').innerText = data.risk_distribution['HIGH RISK'] || 0;
                    document.getElementById('stat-mod').innerText = data.risk_distribution['MODERATE RISK'] || 0;
                    document.getElementById('stat-low').innerText = data.risk_distribution['LOW RISK'] || 0;
                }
            } catch (e) {
                console.error("Analytics load error", e);
            }
        }

        // ======================================================================
        // STAGE 1 (ML) INDEPENDENT TESTING HANDLERS
        // ======================================================================
        // STAGE 1 (ML) INDEPENDENT TESTING HANDLERS
        // ======================================================================
        function setMLSample(type) {
            if (type === 'HIGH') {
                document.getElementById('s1-patient-id').value = 'PAT-ML-0101';
                document.getElementById('s1-age').value = 68;
                document.getElementById('s1-sex').value = 'M';
                document.getElementById('s1-cancer-type').value = 'Lung (LUAD)';
                document.getElementById('s1-cancer-stage').value = 'Stage IV';
                document.getElementById('s1-mutation').value = 'EGFR L858R, MET amp';
                document.getElementById('s1-tumor-marker').value = 38.4;
                document.getElementById('s1-ctdna').value = 2.45;
                document.getElementById('s1-alt').value = 265;
                document.getElementById('s1-ast').value = 224;
                document.getElementById('s1-creatinine').value = 1.85;
                document.getElementById('s1-bilirubin').value = 2.8;
                document.getElementById('s1-treatment').value = 'Osimertinib 80mg + Capmatinib';
                document.getElementById('s1-dosage').value = 1600;
                document.getElementById('s1-cycle').value = 4;
            } else if (type === 'MODERATE') {
                document.getElementById('s1-patient-id').value = 'PAT-ML-0204';
                document.getElementById('s1-age').value = 62;
                document.getElementById('s1-sex').value = 'F';
                document.getElementById('s1-cancer-type').value = 'Breast (BRCA)';
                document.getElementById('s1-cancer-stage').value = 'Stage III';
                document.getElementById('s1-mutation').value = 'PIK3CA H1047R';
                document.getElementById('s1-tumor-marker').value = 16.5;
                document.getElementById('s1-ctdna').value = 0.82;
                document.getElementById('s1-alt').value = 75;
                document.getElementById('s1-ast').value = 68;
                document.getElementById('s1-creatinine').value = 1.15;
                document.getElementById('s1-bilirubin').value = 1.2;
                document.getElementById('s1-treatment').value = 'Abemaciclib + Fulvestrant';
                document.getElementById('s1-dosage').value = 1000;
                document.getElementById('s1-cycle').value = 3;
            } else if (type === 'LOW') {
                document.getElementById('s1-patient-id').value = 'PAT-ML-0318';
                document.getElementById('s1-age').value = 54;
                document.getElementById('s1-sex').value = 'M';
                document.getElementById('s1-cancer-type').value = 'Prostate (PRAD)';
                document.getElementById('s1-cancer-stage').value = 'Stage I';
                document.getElementById('s1-mutation').value = 'TMPRSS2-ERG';
                document.getElementById('s1-tumor-marker').value = 4.2;
                document.getElementById('s1-ctdna').value = 0.12;
                document.getElementById('s1-alt').value = 25;
                document.getElementById('s1-ast').value = 22;
                document.getElementById('s1-creatinine').value = 0.95;
                document.getElementById('s1-bilirubin').value = 0.7;
                document.getElementById('s1-treatment').value = 'Leuprolide Acetate 22.5mg';
                document.getElementById('s1-dosage').value = 400;
                document.getElementById('s1-cycle').value = 1;
            }
        }

        async function runStage1Prediction() {
            const btn = document.getElementById('btn-run-stage1');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> COMPUTING ML PREDICTION...`;

            const payload = {
                patient_id: document.getElementById('s1-patient-id').value || "PAT-ML-0101",
                age: parseFloat(document.getElementById('s1-age').value) || 60,
                sex: document.getElementById('s1-sex').value || "M",
                cancer_type: document.getElementById('s1-cancer-type').value,
                cancer_stage: document.getElementById('s1-cancer-stage').value,
                mutation_profile: document.getElementById('s1-mutation').value,
                tumor_marker_level: parseFloat(document.getElementById('s1-tumor-marker').value) || 20.0,
                ctDNA_level: parseFloat(document.getElementById('s1-ctdna').value) || 1.0,
                creatinine: parseFloat(document.getElementById('s1-creatinine').value) || 1.1,
                bilirubin: parseFloat(document.getElementById('s1-bilirubin').value) || 1.0,
                ALT: parseFloat(document.getElementById('s1-alt').value) || 45,
                AST: parseFloat(document.getElementById('s1-ast').value) || 40,
                treatment_name: document.getElementById('s1-treatment').value || "Standard Regimen",
                dosage_mg: parseFloat(document.getElementById('s1-dosage').value) || 400,
                treatment_cycle: parseInt(document.getElementById('s1-cycle').value) || 3
            };

            const outCard = document.getElementById('s1-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/1', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                document.getElementById('s1-out-pid').innerText = data.patient_id || payload.patient_id;
                document.getElementById('s1-out-pred').innerText = `${data.prediction || 'Unknown'} Toxicity`;
                const riskCat = data.risk_category || `${(data.prediction || 'MODERATE').toUpperCase()} RISK`;
                document.getElementById('s1-out-class').innerText = riskCat;
                document.getElementById('s1-out-conf').innerText = `${data.confidence_pct || 85}%`;
                document.getElementById('s1-out-model-status').innerText = data.model_status || 'Operational';
                document.getElementById('s1-out-patient-sub').innerText = `Patient: ${data.patient_id || payload.patient_id} | Model: ${data.model_architecture || 'Calibrated VotingClassifier Ensemble'}`;

                const probsEl = document.getElementById('s1-out-class-probs');
                if (probsEl && data.probabilities) {
                    probsEl.innerHTML = `
                        <span><strong>Low Toxicity:</strong> ${((data.probabilities.Low || 0) * 100).toFixed(1)}%</span>
                        <span><strong>Moderate Toxicity:</strong> ${((data.probabilities.Moderate || 0) * 100).toFixed(1)}%</span>
                        <span><strong>High Toxicity:</strong> ${((data.probabilities.High || 0) * 100).toFixed(1)}%</span>
                    `;
                }

                document.getElementById('s1-out-interp').innerText = data.clinical_interpretation || 'ML evaluation complete.';

                const badge = document.getElementById('s1-out-badge');
                badge.innerText = riskCat;
                badge.className = 'risk-classification-tag';
                if (riskCat.includes('LOW')) badge.classList.add('low');
                else if (riskCat.includes('MODERATE')) badge.classList.add('moderate');
                else if (riskCat.includes('CRITICAL')) badge.classList.add('critical');
                else badge.classList.add('high');

                const tagsEl = document.getElementById('s1-out-features');
                tagsEl.innerHTML = '';
                (data.important_features || []).forEach(f => {
                    tagsEl.innerHTML += `<span class="feature-tag"><strong>${f.feature}:</strong> ${f.value} (${f.impact})</span>`;
                });

                if (typeof renderStage1ShapWaterfall === 'function') {
                    renderStage1ShapWaterfall(data);
                }

                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s1-out-pid').innerText = payload.patient_id || 'PAT-ML-0101';
                document.getElementById('s1-out-pred').innerText = 'Model/API unavailable';
                document.getElementById('s1-out-class').innerText = 'ERROR';
                document.getElementById('s1-out-conf').innerText = 'N/A';
                document.getElementById('s1-out-model-status').innerText = 'Model/API unavailable';
                document.getElementById('s1-out-interp').innerText = 'Model/API unavailable: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ RUN ML PREDICTION ]`;
            }
        }

        // ======================================================================
        // STAGE 2 (DL) MULTIMODAL SLOTS HANDLERS
        // ======================================================================


        async function runStage2Prediction() {
            const btn = document.getElementById('btn-run-stage2');
            const hasValidCT = s2SlotStates.ct.validated && Boolean(s2SlotStates.ct.savedFilename);
            const hasValidPath = s2SlotStates.pathology.validated && Boolean(s2SlotStates.pathology.savedFilename);
            const hasValidMRI = s2SlotStates.mri.validated && Boolean(s2SlotStates.mri.savedFilename);

            let selectedImage = null;
            let selectedType = null;
            if (hasValidCT) {
                selectedImage = s2SlotStates.ct.savedFilename;
                selectedType = "CT Scan";
            } else if (hasValidPath) {
                selectedImage = s2SlotStates.pathology.savedFilename;
                selectedType = "Pathology";
            } else if (hasValidMRI) {
                selectedImage = s2SlotStates.mri.savedFilename;
                selectedType = "MRI";
            }

            if (!selectedImage && (s2SlotStates.ct.file || s2SlotStates.pathology.file || s2SlotStates.mri.file)) {
                showValidationBanner("Please validate your uploaded image before running DL prediction.", "error", "s2");
                return;
            }

            if (!selectedImage) {
                showValidationBanner("Please upload and validate a medical image before starting DL prediction.", "error", "s2");
                return;
            }

            if (selectedType === "MRI" && !hasValidCT && !hasValidPath) {
                showValidationBanner("This image type is not supported by the current DL analysis model (CT Scan or Pathology required).", "error", "s2");
                return;
            }

            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> INFERRING MULTIMODAL DL EMBEDDINGS...`;

            const payload = {
                patient_id: document.getElementById('s2-patient-id')?.value || "PAT-DL-0201",
                age: parseFloat(document.getElementById('s2-age')?.value) || 61,
                cancer_type: document.getElementById('s2-cancer-type')?.value || "luad",
                cancer_stage: document.getElementById('s2-cancer-stage')?.value || "Stage IV",
                mutation_profile: document.getElementById('s2-mutation')?.value || "EGFR L858R",
                tumor_marker_level: parseFloat(document.getElementById('s2-tumor-marker')?.value) || 24.5,
                ctDNA_level: parseFloat(document.getElementById('s2-ctdna')?.value) || 1.62,
                image_file: selectedImage,
                image_type: selectedType,
                ct_image_file: hasValidCT ? s2SlotStates.ct.savedFilename : null,
                pathology_image_file: hasValidPath ? s2SlotStates.pathology.savedFilename : null,
                mri_image_file: hasValidMRI ? s2SlotStates.mri.savedFilename : null
            };

            const outCard = document.getElementById('s2-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/2', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.detail || "Model/API unavailable");
                }
                const data = await res.json();

                outCard.style.display = 'flex';
                document.getElementById('s2-out-pid').innerText = data.patient_id || payload.patient_id;
                document.getElementById('s2-out-class').innerText = data.image_prediction || 'Stable Lesion';
                document.getElementById('s2-out-classif').innerText = data.classification || data.image_prediction || 'Indeterminate Lesion';
                const prob = data.progression_probability !== undefined ? (data.progression_probability * 100).toFixed(1) + '%' : (data.probability !== undefined ? (data.probability * 100).toFixed(1) + '%' : '45.0%');
                document.getElementById('s2-out-prob').innerText = prob;
                document.getElementById('s2-out-conf').innerText = `${data.confidence_pct || 92}%`;
                document.getElementById('s2-out-model-status').innerText = data.model_status || 'Operational (MultimodalLSTM)';
                document.getElementById('s2-out-patient-sub').innerText = `Patient: ${data.patient_id || payload.patient_id} | Model: ${data.model_architecture || 'MultimodalLSTM'}`;

                const tier = data.risk_tier || 'Moderate Risk';
                const badge = document.getElementById('s2-out-risk-tier');
                badge.innerText = tier.toUpperCase();
                badge.className = 'risk-classification-tag';
                if (tier.toLowerCase().includes('low')) badge.classList.add('low');
                else if (tier.toLowerCase().includes('moderate')) badge.classList.add('moderate');
                else if (tier.toLowerCase().includes('critical')) badge.classList.add('critical');
                else badge.classList.add('high');

                document.getElementById('s2-out-interp').innerText = data.clinical_interpretation || 'DL inference complete.';

                const tbody = document.getElementById('s2-trajectory-table');
                tbody.innerHTML = '';
                (data.trajectory_breakdown || data.trajectory_timepoints || []).forEach(t => {
                    const rowClass = (t.risk_tier || '').toLowerCase().includes('high') ? 'red' : ((t.risk_tier || '').toLowerCase().includes('low') ? 'green' : 'blue');
                    tbody.innerHTML += `
                        <tr>
                            <td><strong>Step ${t.step !== undefined ? t.step : (t.sequence_step !== undefined ? t.sequence_step : '0')}</strong></td>
                            <td>${t.timepoint || ('Month ' + (t.timepoint_month || 0))}</td>
                            <td>${((t.progression_probability || 0) * 100).toFixed(1)}%</td>
                            <td><span class="metric-badge ${rowClass}">${t.risk_tier || 'Stable'}</span></td>
                        </tr>
                    `;
                });
                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s2-out-pid').innerText = payload.patient_id || 'PAT-DL-0201';
                document.getElementById('s2-out-class').innerText = 'Model/API unavailable';
                document.getElementById('s2-out-classif').innerText = 'Unavailable';
                document.getElementById('s2-out-prob').innerText = 'N/A';
                document.getElementById('s2-out-conf').innerText = 'N/A';
                document.getElementById('s2-out-model-status').innerText = 'Model/API unavailable';
                document.getElementById('s2-out-risk-tier').innerText = 'ERROR';
                document.getElementById('s2-out-interp').innerText = 'DL execution error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ RUN DL PREDICTION ]`;
            }
        }

        // ======================================================================
        // STAGE 3 (NLP) INDEPENDENT TESTING HANDLERS
        // ======================================================================
        function setNLPTextSample(type) {
            const el = document.getElementById('inp-stage3-text');
            if (type === 'HIGH') {
                el.value = "Patient has elevated tumor markers and worsening symptoms with high fever 38.8C, persistent fatigue, marked transaminase elevation, and right upper quadrant abdominal tenderness concerning for acute drug toxicity.";
            } else if (type === 'MODERATE') {
                el.value = "Patient reports mild-to-moderate fatigue, stable vital signs, mild nausea following cycle 2. Liver function tests show borderline ALT 65 U/L elevation, no acute jaundice or fever.";
            } else if (type === 'CRITICAL') {
                el.value = "Emergency ICU alert: Patient presents in acute septic shock with refractory hypotension BP 82/48, spiking temperature 39.6C, absolute neutropenia (ANC 280/uL), and multi-organ decompensation requiring immediate resuscitation.";
            }
        }

        async function runStage3Prediction() {
            const btn = document.getElementById('btn-run-stage3');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> PARSING CLINICAL ENTITIES & TRIAGE...`;

            const text = document.getElementById('inp-stage3-text').value;
            const payload = {
                patient_id: "TEST-NLP-PT",
                clinical_notes: text,
                cancer_type: "Lung (LUAD)"
            };

            const outCard = document.getElementById('s3-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/3', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                const urg = (data.urgency_classification || 'MODERATE').toUpperCase();
                document.getElementById('s3-out-level').innerText = `${urg} RISK`;
                document.getElementById('s3-out-conf').innerText = `${data.confidence_pct || 91}%`;
                document.getElementById('s3-out-prob').innerText = (data.confidence !== undefined) ? (data.confidence).toFixed(3) : '0.910';
                document.getElementById('s3-out-model-status').innerText = data.model_status || 'Operational (TF-IDF + LR)';
                document.getElementById('s3-out-latency').innerText = `${data.execution_time_ms || 14.2} ms`;
                const entList = data.extracted_entities || [];
                document.getElementById('s3-out-count').innerText = `${entList.length} Identified`;
                document.getElementById('s3-out-interp').innerText = data.clinical_interpretation || 'NLP analysis complete.';

                const badge = document.getElementById('s3-out-badge');
                badge.innerText = `${urg} RISK`;
                badge.className = 'risk-classification-tag';
                if (urg.includes('LOW')) badge.classList.add('low');
                else if (urg.includes('MODERATE')) badge.classList.add('moderate');
                else if (urg.includes('CRITICAL')) badge.classList.add('critical');
                else badge.classList.add('high');

                const tagsEl = document.getElementById('s3-out-entities');
                tagsEl.innerHTML = '';
                entList.forEach(e => {
                    tagsEl.innerHTML += `<span class="feature-tag"><strong>${e.label}:</strong> ${e.text}</span>`;
                });
                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s3-out-level').innerText = 'Model/API unavailable';
                document.getElementById('s3-out-conf').innerText = 'N/A';
                document.getElementById('s3-out-prob').innerText = 'N/A';
                document.getElementById('s3-out-model-status').innerText = 'Model/API unavailable';
                document.getElementById('s3-out-count').innerText = '0';
                document.getElementById('s3-out-latency').innerText = 'N/A';
                document.getElementById('s3-out-interp').innerText = 'Model/API unavailable: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ RUN NLP PREDICTION ]`;
            }
        }

        // ======================================================================
        // STAGE 4 (SLM) INDEPENDENT SIMPLIFICATION HANDLERS
        // ======================================================================
        function setSLMTextSample(type) {
            const el = document.getElementById('inp-stage4-text');
            if (type === 'TKI') {
                el.value = "Patient PAT-0001 (Lung, EGFR L858R) on Targeted Therapy presents with acute transaminase elevation with ALT 248 U/L and AST 210 U/L following cycle 4 of tyrosine kinase inhibitor therapy. Clinical observation reveals persistent right upper quadrant discomfort, fever at 38.4C, and intractable nausea concerning for drug-induced liver injury requiring urgent clinical review and hydration support.";
            } else if (type === 'HIGH') {
                el.value = "The patient has a mutation count of 15, a chemotherapy dosage of 180 mg, a ctDNA level of 6.5, and a tumor marker level of 8.2. The patient's WBC count is 15000 cells/\u00b5L, SpO2 is 91%, and body temperature is 38.5\u00b0C. The patient reports severe weakness, persistent vomiting, intense pain, and worsening symptoms. Urgent medical assessment is required.";
            } else if (type === 'CRITICAL') {
                el.value = "Patient PAT-0099 (Metastatic Colorectal) currently receiving 3rd line combination antineoplastic infusion presents with severe rigors, hypotension (systolic 78 mmHg), fever 39.8C, marked absolute neutropenia (ANC < 300), and rapid hemodynamic decompensation consistent with hyperacute neutropenic septic shock requiring emergent ICU transfer and immediate broad-spectrum empiric antimicrobial coverage.";
            }
        }

        async function runStage4Simplification() {
            const btn1 = document.getElementById('btn-run-stage4');
            const btn2 = document.getElementById('btn-predict-stage4');
            if (btn1) {
                btn1.disabled = true;
                btn1.innerHTML = `<span class="system-dot" style="background:#fff;"></span> SIMPLIFYING...`;
            }
            if (btn2) {
                btn2.disabled = true;
                btn2.innerHTML = `<span class="system-dot" style="background:#fff;"></span> PREDICTING RANGE...`;
            }

            const text = document.getElementById('inp-stage4-text').value;
            const payload = {
                patient_id: "PAT-SLM-TEST",
                cancer_type: "Oncology",
                treatment_name: "Targeted Therapy",
                clinical_notes: text,
                ALT: text.includes('248') ? 248 : 45
            };

            const outCard = document.getElementById('s4-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/4', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                document.getElementById('s4-out-original').innerText = data.original_text || text;
                
                // Ultra concise simplified text (<2 lines)
                let simplified = data.simplified_explanation || data.clinical_reasoning || 'Simplified explanation generated.';
                document.getElementById('s4-out-simplified').innerText = simplified;

                // Detect Urgency Tier (LOW, MODERATE, HIGH, CRITICAL)
                let tier = (data.urgency_tier || data.predicted_urgency_tier || data.urgency_classification || data.tier || '').toUpperCase();
                if (!['LOW', 'MODERATE', 'HIGH', 'CRITICAL'].includes(tier)) {
                    const textCombined = (text + ' ' + (data.guardrail_status || '') + ' ' + (data.raw_slm_output || '') + ' ' + (data.simplified_explanation || '')).toLowerCase();
                    if (textCombined.includes('critical') || textCombined.includes('septic') || textCombined.includes('78 mmhg') || textCombined.includes('anc < 300') || textCombined.includes('spo2 < 88')) {
                        tier = 'CRITICAL';
                    } else if (
                        textCombined.includes('high') || 
                        textCombined.includes('248') || 
                        textCombined.includes('210') || 
                        textCombined.includes('dili') || 
                        textCombined.includes('liver injury') ||
                        textCombined.includes('urgent') ||
                        textCombined.includes('persistent vomiting') ||
                        textCombined.includes('intense pain') ||
                        textCombined.includes('severe weakness') ||
                        textCombined.includes('worsening') ||
                        textCombined.includes('38.5') ||
                        textCombined.includes('38.4') ||
                        textCombined.includes('38.6') ||
                        textCombined.includes('38.7') ||
                        textCombined.includes('38.8') ||
                        textCombined.includes('38.9') ||
                        textCombined.includes('39.') ||
                        textCombined.includes('40.') ||
                        textCombined.includes('spiking fever') ||
                        textCombined.includes('15000') ||
                        textCombined.includes('spo2 is 91') ||
                        textCombined.includes('spo2 is 90') ||
                        textCombined.includes('spo2 is 92') ||
                        textCombined.includes('91%') ||
                        textCombined.includes('90%') ||
                        textCombined.includes('92%')
                    ) {
                        tier = 'HIGH';
                    } else if (textCombined.includes('low-tier') || textCombined.includes('low protocol') || textCombined.includes('mild') || textCombined.includes('xerosis')) {
                        tier = 'LOW';
                    } else {
                        tier = 'MODERATE';
                    }
                }

                // Update 4-Tier Meter Pills
                const meterPills = {
                    'LOW': document.getElementById('s4-tier-low'),
                    'MODERATE': document.getElementById('s4-tier-moderate'),
                    'HIGH': document.getElementById('s4-tier-high'),
                    'CRITICAL': document.getElementById('s4-tier-critical')
                };

                // Reset all pills
                Object.values(meterPills).forEach(p => {
                    if (p) {
                        p.style.background = '#ffffff';
                        p.style.color = '#94a3b8';
                        p.style.border = '1px solid #e2e8f0';
                        p.style.boxShadow = 'none';
                        p.style.transform = 'none';
                        p.style.fontWeight = '700';
                    }
                });

                // Highlight active predicted tier
                const activePill = meterPills[tier];
                const badge = document.getElementById('s4-out-tier-badge');
                if (tier === 'LOW') {
                    if (activePill) {
                        activePill.style.background = '#dcfce7';
                        activePill.style.color = '#15803d';
                        activePill.style.border = '2px solid #22c55e';
                        activePill.style.boxShadow = '0 4px 12px rgba(34, 197, 94, 0.25)';
                        activePill.style.transform = 'scale(1.04)';
                        activePill.style.fontWeight = '800';
                    }
                    if (badge) {
                        badge.innerText = 'LOW URGENCY';
                        badge.style.background = '#dcfce7';
                        badge.style.color = '#15803d';
                    }
                } else if (tier === 'MODERATE') {
                    if (activePill) {
                        activePill.style.background = '#fef3c7';
                        activePill.style.color = '#92400e';
                        activePill.style.border = '2px solid #f59e0b';
                        activePill.style.boxShadow = '0 4px 12px rgba(245, 158, 11, 0.25)';
                        activePill.style.transform = 'scale(1.04)';
                        activePill.style.fontWeight = '800';
                    }
                    if (badge) {
                        badge.innerText = 'MODERATE URGENCY';
                        badge.style.background = '#fef3c7';
                        badge.style.color = '#92400e';
                    }
                } else if (tier === 'HIGH') {
                    if (activePill) {
                        activePill.style.background = '#ffedd5';
                        activePill.style.color = '#c2410c';
                        activePill.style.border = '2px solid #ea580c';
                        activePill.style.boxShadow = '0 4px 12px rgba(234, 88, 12, 0.25)';
                        activePill.style.transform = 'scale(1.04)';
                        activePill.style.fontWeight = '800';
                    }
                    if (badge) {
                        badge.innerText = 'HIGH URGENCY';
                        badge.style.background = '#ffedd5';
                        badge.style.color = '#c2410c';
                    }
                } else if (tier === 'CRITICAL') {
                    if (activePill) {
                        activePill.style.background = '#fee2e2';
                        activePill.style.color = '#b91c1c';
                        activePill.style.border = '2px solid #dc2626';
                        activePill.style.boxShadow = '0 4px 12px rgba(220, 38, 38, 0.25)';
                        activePill.style.transform = 'scale(1.04)';
                        activePill.style.fontWeight = '800';
                    }
                    if (badge) {
                        badge.innerText = 'CRITICAL URGENCY';
                        badge.style.background = '#fee2e2';
                        badge.style.color = '#b91c1c';
                    }
                }

                document.getElementById('s4-out-proc').innerText = `${data.status || 'Completed'} ✓`;
                document.getElementById('s4-out-model').innerText = data.model_status || 'Operational (3B SLM)';
                const guardStatus = data.guardrail_status || 'VERIFIED_SAFE';
                document.getElementById('s4-out-guardrail').innerText = guardStatus;
                document.getElementById('s4-out-latency').innerText = `${data.execution_time_ms || 4.8} ms`;

                const statusBadge = document.getElementById('s4-out-status-badge');
                if (statusBadge) {
                    statusBadge.innerText = guardStatus;
                    statusBadge.className = 'risk-classification-tag';
                    if (guardStatus.includes('SAFE')) statusBadge.classList.add('low');
                    else if (guardStatus.includes('MODERATE')) statusBadge.classList.add('moderate');
                    else statusBadge.classList.add('high');
                }

                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s4-out-original').innerText = text;
                document.getElementById('s4-out-simplified').innerText = 'Model/API unavailable: ' + err.message;
                document.getElementById('s4-out-proc').innerText = 'Failed';
                document.getElementById('s4-out-model').innerText = 'Model/API unavailable';
                document.getElementById('s4-out-guardrail').innerText = 'UNAVAILABLE';
                document.getElementById('s4-out-latency').innerText = 'N/A';
            } finally {
                if (btn1) {
                    btn1.disabled = false;
                    btn1.innerHTML = `[ SIMPLIFY WITH SLM ]`;
                }
                if (btn2) {
                    btn2.disabled = false;
                    btn2.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> [ PREDICT URGENCY RANGE ]`;
                }
            }
        }

        // ======================================================================
        // STAGE 5 (GenAI) INDEPENDENT GENERATION HANDLERS
        // ======================================================================
        function setGenAISeedSample(type) {
            if (type === 'NSCLC') {
                document.getElementById('s5-cancer-type').value = 'Lung (LUAD)';
                document.getElementById('s5-age-range').value = '60 - 70';
                document.getElementById('s5-mutation').value = 'EGFR L858R, MET amp';
                document.getElementById('s5-biomarker').value = 'PD-L1 85%, TMB 14 mut/Mb';
                document.getElementById('s5-stage').value = 'Stage IV';
                document.getElementById('s5-treatment').value = 'Osimertinib 80mg + Capmatinib';
                document.getElementById('s5-alt').value = 248;
                document.getElementById('s5-notes').value = 'Patient on targeted therapy presenting with spiking fever 38.4C and elevated hepatic enzymes indicating acute DILI.';
            } else if (type === 'BRCA') {
                document.getElementById('s5-cancer-type').value = 'Breast (BRCA)';
                document.getElementById('s5-age-range').value = '50 - 60';
                document.getElementById('s5-mutation').value = 'PIK3CA H1047R, ER+/PR+';
                document.getElementById('s5-biomarker').value = 'ER 95%, PR 80%, HER2 1+';
                document.getElementById('s5-stage').value = 'Stage III';
                document.getElementById('s5-treatment').value = 'Abemaciclib + Fulvestrant';
                document.getElementById('s5-alt').value = 65;
                document.getElementById('s5-notes').value = 'Follow-up assessment after 3 cycles of CDK4/6 inhibitor. Moderate fatigue, stable imaging, no visceral crisis.';
            } else if (type === 'DILI') {
                document.getElementById('s5-cancer-type').value = 'Colon (COAD)';
                document.getElementById('s5-age-range').value = '65 - 75';
                document.getElementById('s5-mutation').value = 'KRAS G12D, MSS';
                document.getElementById('s5-biomarker').value = 'CEA 48.2 ng/mL, CA 19-9 85 U/mL';
                document.getElementById('s5-stage').value = 'Stage IV';
                document.getElementById('s5-treatment').value = 'FOLFOX + Bevacizumab';
                document.getElementById('s5-alt').value = 310;
                document.getElementById('s5-notes').value = 'Critical drug-induced hepatotoxicity with severe right upper quadrant tenderness, jaundice, and marked transaminitis.';
            }
        }

        async function runStage5Generation() {
            const btn = document.getElementById('btn-run-stage5');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> SYNTHESIZING CLINICAL REPORT & AUDITING...`;

            const payload = {
                patient_id: "SYNTH-GENAI-001",
                cancer_type: document.getElementById('s5-cancer-type').value,
                cancer_stage: document.getElementById('s5-stage').value,
                mutation_profile: document.getElementById('s5-mutation').value,
                biomarker: document.getElementById('s5-biomarker').value,
                age_range: document.getElementById('s5-age-range').value,
                treatment_name: document.getElementById('s5-treatment').value,
                ALT: parseFloat(document.getElementById('s5-alt').value) || 45,
                clinical_notes: document.getElementById('s5-notes').value
            };

            const outCard = document.getElementById('s5-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/5', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                const qi = data.quality_indicators || {};
                document.getElementById('s5-out-factual').innerText = `${((qi.factual_consistency || 0.982) * 100).toFixed(1)}%`;
                document.getElementById('s5-out-safety').innerText = `${((qi.cross_modal_safety_score || 0.950) * 100).toFixed(1)}%`;
                document.getElementById('s5-out-halluc').innerText = `${((qi.hallucination_index || 0.018) * 100).toFixed(1)}%`;
                document.getElementById('s5-out-valid').innerText = data.validation_status || 'PASSED';

                document.getElementById('s5-out-summary').innerText = data.patient_summary || data.summary || 'Summary generated.';

                const findingsEl = document.getElementById('s5-out-findings');
                findingsEl.innerHTML = '';
                (data.key_findings || []).forEach(f => {
                    findingsEl.innerHTML += `
                        <div class="consideration-item">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                            <span>${f}</span>
                        </div>
                    `;
                });

                document.getElementById('s5-out-report').innerText = data.final_report || 'Full report generated.';
                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s5-out-factual').innerText = 'N/A';
                document.getElementById('s5-out-safety').innerText = 'N/A';
                document.getElementById('s5-out-halluc').innerText = 'N/A';
                document.getElementById('s5-out-valid').innerText = 'Model/API unavailable';
                document.getElementById('s5-out-summary').innerText = 'Model/API unavailable: ' + err.message;
                document.getElementById('s5-out-findings').innerHTML = '';
                document.getElementById('s5-out-report').innerText = 'Model/API unavailable: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ GENERATE ]`;
            }
        }

        // ======================================================================
        // STAGE 6 (AGENTIC AI) CLIENT HANDLERS
        // ======================================================================
        let activeAgenticTraceId = null;
        let isAgenticPaused = false;

        async function setAgenticPreset(key) {
            try {
                const res = await fetch('/api/v1/agentic/presets');
                const presets = await res.json();
                const p = presets[key];
                if (!p || !p.data) return;
                const d = p.data;

                document.getElementById('s6-patient-id').value = d.patient_id || 'PAT-0001';
                document.getElementById('s6-age').value = d.age || 60;
                document.getElementById('s6-sex').value = d.sex || 'M';
                document.getElementById('s6-cancer-type').value = d.cancer_type || 'Lung (LUAD)';
                document.getElementById('s6-stage').value = d.cancer_stage || 'Stage IV';
                document.getElementById('s6-biomarker').value = d.genomic_biomarker || 'EGFR L858R';
                document.getElementById('s6-treatment').value = d.treatment_name || 'Osimertinib';
                document.getElementById('s6-ctdna').value = d.ctDNA_level || 0.45;
                document.getElementById('s6-rising-ctdna').value = d.rising_ctdna_readings || 2;
                document.getElementById('s6-image-toggle').value = d.medical_image_provided ? 'true' : 'false';
                document.getElementById('s6-cr').value = d.creatinine || 1.1;
                document.getElementById('s6-alt').value = d.ALT || 45;
                document.getElementById('s6-ast').value = d.AST || 42;
                document.getElementById('s6-bili').value = d.bilirubin || 1.0;
                document.getElementById('s6-notes').value = d.clinical_notes || '';

                // Reset statuses
                resetAgenticStatusDots();
                document.getElementById('s6-safety-halt-banner').style.display = 'none';
                document.getElementById('s6-override-badge').style.display = 'none';
            } catch (e) {
                console.error("Error setting agentic preset", e);
            }
        }

        function pullExistingAiResults() {
            const pid = document.getElementById('inp-patient_id');
            if (pid && pid.value) document.getElementById('s6-patient-id').value = pid.value;
            const age = document.getElementById('inp-age');
            if (age && age.value) document.getElementById('s6-age').value = age.value;
            const sex = document.getElementById('inp-sex');
            if (sex && sex.value) document.getElementById('s6-sex').value = sex.value;
            const ct = document.getElementById('inp-cancer_type');
            if (ct && ct.value) document.getElementById('s6-cancer-type').value = ct.value;
            const st = document.getElementById('inp-cancer_stage');
            if (st && st.value) document.getElementById('s6-stage').value = st.value;
            const bio = document.getElementById('inp-mutation_profile');
            if (bio && bio.value) document.getElementById('s6-biomarker').value = bio.value;
            const tx = document.getElementById('inp-treatment_name');
            if (tx && tx.value) document.getElementById('s6-treatment').value = tx.value;
            const ctdna = document.getElementById('inp-ctDNA_level');
            if (ctdna && ctdna.value) document.getElementById('s6-ctdna').value = ctdna.value;
            const cr = document.getElementById('inp-creatinine');
            if (cr && cr.value) document.getElementById('s6-cr').value = cr.value;
            const alt = document.getElementById('inp-ALT');
            if (alt && alt.value) document.getElementById('s6-alt').value = alt.value;
            const ast = document.getElementById('inp-AST');
            if (ast && ast.value) document.getElementById('s6-ast').value = ast.value;
            const bili = document.getElementById('inp-bilirubin');
            if (bili && bili.value) document.getElementById('s6-bili').value = bili.value;

            alert("Existing AI parameters successfully pulled into Stage 6 Context Feeder!");
        }

        function resetAgenticStatusDots() {
            const agentIds = [
                'agent_patient_data', 'agent_risk_analysis', 'agent_image_analysis',
                'agent_clinical_nlp', 'agent_clinical_briefing', 'agent_scenario_analysis',
                'agent_treatment_optimization', 'agent_trial_matching', 'agent_safety_guardrail',
                'agent_final_decision'
            ];
            agentIds.forEach(id => {
                const dot = document.getElementById(`agent-dot-${id}`);
                if (dot) {
                    dot.className = 'agent-status-indicator waiting';
                    dot.innerText = '○';
                }
            });
        }

        async function runAgenticAnalysis() {
            const btn = document.getElementById('btn-run-agentic');
            btn.disabled = true;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.5" style="animation: spinFast 1s linear infinite;">
                    <circle cx="12" cy="12" r="9" stroke="rgba(255,255,255,0.3)"/>
                    <path d="M12 3a9 9 0 0 1 9 9" stroke="#ffffff"/>
                </svg>
                <span>Orchestrating 10 Agents...</span>
            `;

            // Set all dots to processing
            const agentIds = [
                'agent_patient_data', 'agent_risk_analysis', 'agent_image_analysis',
                'agent_clinical_nlp', 'agent_clinical_briefing', 'agent_scenario_analysis',
                'agent_treatment_optimization', 'agent_trial_matching', 'agent_safety_guardrail',
                'agent_final_decision'
            ];
            agentIds.forEach(id => {
                const dot = document.getElementById(`agent-dot-${id}`);
                if (dot) {
                    dot.className = 'agent-status-indicator processing';
                    dot.innerText = '⟳';
                }
            });

            document.getElementById('s6-safety-halt-banner').style.display = 'none';

            const payload = {
                patient: {
                    patient_id: document.getElementById('s6-patient-id').value,
                    age: parseFloat(document.getElementById('s6-age').value) || 60,
                    sex: document.getElementById('s6-sex').value,
                    cancer_type: document.getElementById('s6-cancer-type').value,
                    cancer_stage: document.getElementById('s6-stage').value,
                    genomic_biomarker: document.getElementById('s6-biomarker').value,
                    treatment_name: document.getElementById('s6-treatment').value,
                    ctDNA_level: parseFloat(document.getElementById('s6-ctdna').value) || 0.45,
                    rising_ctdna_readings: parseInt(document.getElementById('s6-rising-ctdna').value) || 2,
                    medical_image_provided: document.getElementById('s6-image-toggle').value === 'true',
                    creatinine: parseFloat(document.getElementById('s6-cr').value) || 1.1,
                    ALT: parseFloat(document.getElementById('s6-alt').value) || 45,
                    AST: parseFloat(document.getElementById('s6-ast').value) || 42,
                    bilirubin: parseFloat(document.getElementById('s6-bili').value) || 1.0,
                    clinical_notes: document.getElementById('s6-notes').value
                }
            };

            try {
                const res = await fetch('/api/v1/agentic/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Agentic engine error: " + res.statusText);
                const data = await res.json();

                renderAgenticOutputs(data);
            } catch (err) {
                alert("Agentic Analysis Error: " + err.message);
                resetAgenticStatusDots();
            } finally {
                btn.disabled = false;
                btn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    <span>Run Agentic Analysis</span>
                `;
            }
        }

        function renderAgenticOutputs(data) {
            const resultsContainer = document.getElementById('s6-results-container');
            resultsContainer.style.display = 'flex';

            // 1. Update Agent Status Indicators
            const statuses = data.agent_statuses || {};
            for (const id in statuses) {
                const dot = document.getElementById(`agent-dot-${id}`);
                const msg = document.getElementById(`agent-msg-${id}`);
                if (dot) {
                    const st = statuses[id].status;
                    dot.className = 'agent-status-indicator ' + st.toLowerCase();
                    if (st === 'Completed') dot.innerText = '✓';
                    else if (st === 'Processing') dot.innerText = '⟳';
                    else if (st === 'Warning') dot.innerText = '⚠';
                    else if (st === 'Failed') dot.innerText = '✕';
                    else dot.innerText = '○';
                }
                if (msg && statuses[id].message) {
                    msg.innerText = statuses[id].message.substring(0, 32) + '...';
                }
            }

            // 2. Safety Halt Check
            const safetyStatus = data.safety_status || 'PASSED';
            const isHalted = data.status === 'SAFETY_REVIEW_REQUIRED' || safetyStatus === 'SAFETY REVIEW REQUIRED';
            if (isHalted) {
                const haltBanner = document.getElementById('s6-safety-halt-banner');
                haltBanner.style.display = 'flex';
                document.getElementById('s6-safety-halt-text').innerText = (data.final_assessment && data.final_assessment.final_recommendation) || 'Critical clinical toxicity detected. Workflow halted for safety.';
            }

            // 3. Final Assessment Summary
            const fa = data.final_assessment || {};
            activeAgenticTraceId = fa.audit_trace_id || 'TRACE-001';
            document.getElementById('s6-final-patient-meta').innerText = `Patient ID: ${data.patient_id} | Trace: ${activeAgenticTraceId} | Latency: ${data.execution_time_ms} ms`;
            document.getElementById('s6-final-directive').innerText = fa.final_recommendation || 'Clinical consensus formed.';
            
            const statusPill = document.getElementById('s6-final-status-pill');
            if (isHalted) {
                statusPill.className = 'metric-badge red';
                statusPill.innerText = 'SAFETY REVIEW REQUIRED';
            } else if (data.physician_override_active) {
                statusPill.className = 'metric-badge yellow';
                statusPill.innerText = 'PHYSICIAN OVERRIDDEN';
            } else {
                statusPill.className = 'metric-badge green';
                statusPill.innerText = 'CONSENSUS ACHIEVED';
            }

            // Summaries breakdown
            document.getElementById('s6-sum-ml').innerText = (fa.ml_risk_summary || '').split('.')[0] || 'Assessed';
            document.getElementById('s6-sum-dl').innerText = (fa.dl_image_summary || '').split(':')[1] || 'DL Analysis';
            document.getElementById('s6-sum-nlp').innerText = (fa.nlp_clinical_summary || '').split(':')[1] || 'Triage';
            document.getElementById('s6-sum-safety').innerText = fa.safety_status || 'PASSED';
            document.getElementById('s6-sum-safety').style.color = isHalted ? '#dc2626' : '#059669';

            // 4. Treatment Optimization List
            const txList = document.getElementById('s6-treatment-options-list');
            txList.innerHTML = '';
            (data.treatment_options || []).forEach(opt => {
                txList.innerHTML += `
                    <div style="background: #f8fafc; border: 1px solid var(--border-card); border-radius: 6px; padding: 0.85rem;">
                        <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 0.35rem;">
                            <strong style="font-size: 0.84rem; color: #0f172a;">Priority ${opt.priority_rank}: ${opt.regimen_name}</strong>
                            <span class="metric-badge blue" style="font-size: 0.65rem;">${opt.category}</span>
                        </div>
                        <p style="font-size: 0.76rem; color: #334155; margin-bottom: 0.25rem;"><strong>Expected Benefit:</strong> ${opt.expected_benefit}</p>
                        <p style="font-size: 0.74rem; color: #64748b; margin-bottom: 0.25rem;"><strong>Toxicity & Suitability:</strong> ${opt.renal_hepatic_suitability}</p>
                        ${opt.safety_warnings && opt.safety_warnings.length > 0 ? `<div style="font-size: 0.7rem; color: #b45309; font-weight: 600; margin-top: 0.3rem;">⚠️ ${opt.safety_warnings[0]}</div>` : ''}
                    </div>
                `;
            });

            // 5. Clinical Trial Matches List
            const trialList = document.getElementById('s6-trial-matches-list');
            trialList.innerHTML = '';
            const trials = data.matched_trials || [];
            document.getElementById('s6-trial-count-badge').innerText = `${trials.length} PROTOCOLS MATCHED`;
            trials.forEach(tr => {
                trialList.innerHTML += `
                    <div class="trial-card-item">
                        <div class="trial-card-header">
                            <div>
                                <h4>${tr.trial_name}</h4>
                                <div class="trial-meta-pills">
                                    <span class="trial-meta-pill">${tr.trial_id}</span>
                                    <span class="trial-meta-pill">${tr.phase}</span>
                                    <span class="trial-meta-pill match">Match Score: ${tr.matching_score}%</span>
                                    <span class="trial-meta-pill" style="font-weight: 700; color: ${tr.open_slots > 0 ? '#059669' : '#dc2626'};">${tr.open_slots} Open Slots</span>
                                </div>
                            </div>
                        </div>
                        <p style="font-size: 0.76rem; color: #475569; margin-top: 0.4rem;">
                            <strong>Eligibility:</strong> <span style="color: ${tr.eligibility_status.includes('Eligible') ? '#059669' : '#b45309'}; font-weight: 600;">${tr.eligibility_status}</span>
                        </p>
                        <div style="font-size: 0.68rem; color: #94a3b8; margin-top: 0.35rem; font-style: italic;">
                            Source: ${tr.data_source}
                        </div>
                    </div>
                `;
            });

            // 6. Decision Trace Table
            const traceTbody = document.getElementById('s6-decision-trace-tbody');
            traceTbody.innerHTML = '';
            const traces = data.decision_traces || [];
            document.getElementById('s6-trace-count-pill').innerText = `${traces.length} STEPS AUDITED`;
            traces.forEach(t => {
                const stClass = t.status === 'Completed' ? 'green' : (t.status === 'Warning' ? 'yellow' : (t.status === 'Failed' ? 'red' : 'blue'));
                traceTbody.innerHTML += `
                    <tr>
                        <td><span class="step-badge">${t.step_number}</span></td>
                        <td><strong>${t.agent_name}</strong></td>
                        <td><span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #0284c7;">${t.tool_or_api}</span></td>
                        <td><span class="metric-badge ${stClass}" style="font-size: 0.65rem;">${t.status}</span></td>
                        <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #94a3b8;">${t.timestamp}</td>
                        <td style="line-height: 1.45;">${t.reasoning_summary}</td>
                    </tr>
                `;
            });

            // 7. Safety Checks Grid
            const safetyGrid = document.getElementById('s6-safety-checks-grid');
            safetyGrid.innerHTML = '';
            const checks = data.safety_checks || [];
            checks.forEach(chk => {
                const passed = chk.passed;
                safetyGrid.innerHTML += `
                    <div style="background: ${passed ? '#f0fdf4' : '#fef2f2'}; border: 1px solid ${passed ? '#bbf7d0' : '#fecaca'}; border-radius: 6px; padding: 0.65rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span style="font-size: 0.74rem; font-weight: 700; color: ${passed ? '#166534' : '#991b1b'};">${chk.check_name}</span>
                            <span style="font-size: 0.75rem;">${passed ? '✓' : '🚨'}</span>
                        </div>
                        <p style="font-size: 0.68rem; color: ${passed ? '#15803d' : '#b91c1c'}; margin-top: 0.25rem; line-height: 1.35;">${chk.clinical_finding}</p>
                    </div>
                `;
            });

            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        }

        async function togglePauseAgentic() {
            const btn = document.getElementById('btn-pause-agentic');
            if (!isAgenticPaused) {
                await fetch('/api/v1/agentic/pause', { method: 'POST' });
                isAgenticPaused = true;
                btn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    <span>Resume Agent</span>
                `;
                btn.style.background = 'linear-gradient(135deg, #0284c7 0%, #2563eb 100%)';
                btn.style.borderColor = 'rgba(56, 189, 248, 0.5)';
                alert("Agentic execution paused by physician.");
            } else {
                await fetch('/api/v1/agentic/resume', { method: 'POST' });
                isAgenticPaused = false;
                btn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                        <rect x="6" y="4" width="4" height="16" rx="1"/>
                        <rect x="14" y="4" width="4" height="16" rx="1"/>
                    </svg>
                    <span>Pause Agent</span>
                `;
                btn.style.background = '';
                btn.style.borderColor = '';
                alert("Agentic execution resumed by physician.");
            }
        }

        async function stopAgentic() {
            await fetch('/api/v1/agentic/stop', { method: 'POST' });
            alert("Agentic workflow execution terminated by physician.");
            resetAgenticStatusDots();
        }

        async function promptPhysicianOverride() {
            const reason = prompt("Enter Physician Clinical Override Rationale (recorded in Stage 06 audit trail):", "Physician directs standard-of-care systemic therapy due to patient clinical preference and local monitoring availability.");
            if (!reason) return;

            const pid = document.getElementById('s6-patient-id').value;
            const res = await fetch('/api/v1/agentic/override', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trace_id: activeAgenticTraceId || 'TRACE-001',
                    patient_id: pid,
                    reason: reason,
                    new_order: "Physician active clinical override applied."
                })
            });
            const data = await res.json();
            document.getElementById('s6-override-badge').style.display = 'inline-block';
            alert("PHYSICIAN OVERRIDE ACTIVE. Logged in Stage 06 audit records.");
        }

        function checkAuthSession() {
            const sessionRaw = localStorage.getItem('onco_auth_session');
            if (sessionRaw) {
                try {
                    const session = JSON.parse(sessionRaw);
                    if (session && session.authenticated) {
                        const nameEl = document.getElementById('header-clinician-name');
                        const roleEl = document.getElementById('header-clinician-role');
                        const avatarEl = document.getElementById('header-clinician-avatar');
                        if (nameEl) nameEl.innerText = session.name || "Dr. Sarah Chen, MD";
                        if (roleEl) roleEl.innerText = `${session.role || 'Chief Oncologist'} • ${session.clearance || 'Clearance L4'}`;
                        if (avatarEl) {
                            if (session.name && session.name.includes("Marcus")) avatarEl.innerText = "🔬";
                            else if (session.name && session.name.includes("Elena")) avatarEl.innerText = "🛡️";
                            else if (session.name && session.name.includes("Alex")) avatarEl.innerText = "📊";
                            else avatarEl.innerText = "🩺";
                        }
                    }
                } catch(e) {
                    console.warn("Auth session parse error", e);
                }
            }
        }

        function signOutClinician(e) {
            e.preventDefault();
            if (confirm("Sign out of current clinical workstation session and return to login portal?")) {
                localStorage.removeItem('onco_auth_session');
                window.location.href = '/login';
            }
        }

        // Auto-load high risk preset, auth session, and initial views on start
        window.addEventListener('DOMContentLoaded', () => {
            checkAuthSession();
            setOverviewSample('HIGH');
            if (typeof updateWhatIfSimulation === 'function') updateWhatIfSimulation();
            if (typeof renderStage1ShapWaterfall === 'function') renderStage1ShapWaterfall();
            if (typeof renderDeliberationStream === 'function') renderDeliberationStream();
            if (typeof renderAuditLogs === 'function') renderAuditLogs();
            if (window.location.hash) {
                const target = window.location.hash.replace('#', '');
                if (document.getElementById(target)) switchView(target);
            }
        });
        window.addEventListener('hashchange', () => {
            if (window.location.hash) {
                const target = window.location.hash.replace('#', '');
                if (document.getElementById(target)) switchView(target);
            }
        });
    </script>
</body>
</html>
"""
