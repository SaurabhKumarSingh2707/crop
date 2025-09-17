"""
Demo page to showcase the modern UI improvements
"""

from flask import Flask, render_template_string

app = Flask(__name__)

# Modern comparison template
comparison_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UI Modernization - Before & After</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: #f8fafc;
            padding: 2rem;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #10b981, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 3rem;
        }
        
        .comparison-item {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .comparison-item h2 {
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e5e7eb;
        }
        
        .old {
            border-left: 4px solid #ef4444;
        }
        
        .new {
            border-left: 4px solid #10b981;
        }
        
        .feature-list {
            list-style: none;
        }
        
        .feature-list li {
            padding: 0.5rem 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .feature-list .old li {
            color: #6b7280;
        }
        
        .feature-list .new li {
            color: #374151;
        }
        
        .icon {
            width: 20px;
            text-align: center;
        }
        
        .old .icon {
            color: #ef4444;
        }
        
        .new .icon {
            color: #10b981;
        }
        
        .demo-links {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 2rem;
        }
        
        .demo-link {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 1rem 2rem;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        .demo-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
        }
        
        @media (max-width: 768px) {
            .comparison {
                grid-template-columns: 1fr;
            }
            
            .demo-links {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-seedling"></i> KrishiVannai AI - UI Modernization</h1>
            <p style="color: #6b7280; font-size: 1.1rem;">Before & After Comparison</p>
        </div>
        
        <div class="comparison">
            <div class="comparison-item old">
                <h2><i class="fas fa-times-circle"></i> Old Design</h2>
                <ul class="feature-list old">
                    <li><i class="fas fa-palette icon"></i> Basic gradient background</li>
                    <li><i class="fas fa-square icon"></i> Simple rounded corners</li>
                    <li><i class="fas fa-font icon"></i> Standard system fonts</li>
                    <li><i class="fas fa-image icon"></i> Basic emoji icons</li>
                    <li><i class="fas fa-bars icon"></i> Plain progress bars</li>
                    <li><i class="fas fa-circle icon"></i> Simple loading spinner</li>
                    <li><i class="fas fa-table icon"></i> Basic information layout</li>
                    <li><i class="fas fa-mouse-pointer icon"></i> Standard hover effects</li>
                </ul>
            </div>
            
            <div class="comparison-item new">
                <h2><i class="fas fa-check-circle"></i> Modern Design</h2>
                <ul class="feature-list new">
                    <li><i class="fas fa-magic icon"></i> Modern gradient with multiple colors</li>
                    <li><i class="fas fa-gem icon"></i> Large border radius (24px)</li>
                    <li><i class="fas fa-font icon"></i> Google Fonts (Inter) integration</li>
                    <li><i class="fas fa-icons icon"></i> FontAwesome icon library</li>
                    <li><i class="fas fa-chart-line icon"></i> Animated progress bars with shimmer</li>
                    <li><i class="fas fa-spinner icon"></i> Enhanced spinner with pulse animation</li>
                    <li><i class="fas fa-th icon"></i> CSS Grid-based responsive layout</li>
                    <li><i class="fas fa-hand-pointer icon"></i> Smooth cubic-bezier transitions</li>
                    <li><i class="fas fa-mobile-alt icon"></i> Mobile-responsive design</li>
                    <li><i class="fas fa-paint-brush icon"></i> CSS custom properties (variables)</li>
                    <li><i class="fas fa-eye icon"></i> Modern glassmorphism effects</li>
                    <li><i class="fas fa-rocket icon"></i> Micro-interactions and animations</li>
                </ul>
            </div>
        </div>
        
        <div class="demo-links">
            <a href="http://127.0.0.1:5002" class="demo-link" target="_blank">
                <i class="fas fa-external-link-alt"></i>
                View Modern UI
            </a>
        </div>
        
        <div style="text-align: center; margin-top: 3rem; color: #6b7280;">
            <h3>Key Improvements</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 2rem;">
                <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <i class="fas fa-palette" style="color: #10b981; font-size: 2rem; margin-bottom: 1rem;"></i>
                    <h4>Visual Design</h4>
                    <p>Modern color palette, typography, and spacing</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <i class="fas fa-mobile-alt" style="color: #3b82f6; font-size: 2rem; margin-bottom: 1rem;"></i>
                    <h4>Responsive</h4>
                    <p>Mobile-first design with flexible layouts</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <i class="fas fa-rocket" style="color: #f59e0b; font-size: 2rem; margin-bottom: 1rem;"></i>
                    <h4>Performance</h4>
                    <p>Optimized CSS with hardware acceleration</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def comparison():
    return render_template_string(comparison_template)

if __name__ == '__main__':
    print("🎨 UI Comparison Demo")
    print("Visit: http://127.0.0.1:5003")
    app.run(debug=True, host='127.0.0.1', port=5003)