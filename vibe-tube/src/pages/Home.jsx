
import React from 'react';
import {
  Mic,
  Clapperboard,
  Church,
  MicVocal,
  GitFork,
  Diamond,
  CookingPot,
  Gamepad2,
  Menu,
  Home,
  Tv,
  Users,
  MessageSquare,
  Heart,
  Search,
  Video,
  Bell,
  Power,
  Send,
  Link,
  Copy,
  Share2,
  ThumbsDown,
  Bookmark,
  ChevronDown,
  ChevronLeft, // New import
  ChevronRight, // New import
  Gift, // New import
  Coins, // New import
  Star, // New import
  Music, // New import
  Utensils, // New import
  Globe, // New import
} from 'lucide-react';

const SidebarItem = ({ icon: Icon, label, href = "#", active }) => (
  <a
    href={href}
    className={`flex items-center space-x-4 px-4 py-2.5 text-sm font-medium rounded-lg cursor-pointer transition-all duration-300 ${
      active
        ? 'bg-gradient-to-r from-cyan-500/20 to-pink-500/20 text-cyan-400 border border-cyan-500/30 shadow-lg shadow-cyan-500/20'
        : 'text-cyan-200 hover:bg-gradient-to-r hover:from-cyan-500/10 hover:to-pink-500/10 hover:text-cyan-300 hover:border hover:border-cyan-500/20'
    }`}
  >
    <Icon className="w-4 h-4" />
    <span className="text-sm">{label}</span>
  </a>
);

export default function VtoobeHomePage() {
  const [isTvOn, setIsTvOn] = React.useState(true);
  const [channel, setChannel] = React.useState(3);
  const [volume, setVolume] = React.useState(50);
  const [antennaLeft, setAntennaLeft] = React.useState(-25);
  const [antennaRight, setAntennaRight] = React.useState(15);
  const [showChannel, setShowChannel] = React.useState(false);
  const [nftIndex, setNftIndex] = React.useState(0); // New state for NFT carousel
  const [comments, setComments] = React.useState([
    { id: 1, user: 'TokyoOtaku', comment: 'Kawaii desu! 🌸 Love this aesthetic!', time: '2 min ago' },
    { id: 2, user: 'JakartaVibes', comment: 'Bagus banget! Perfect for late night study 📚', time: '5 min ago' },
    { id: 3, user: 'ManilaGamer', comment: 'Ganda naman! When next karaoke stream? 🎤', time: '8 min ago' },
    { id: 4, user: 'AnimeQueen2024', comment: 'This gives me Studio Ghibli vibes! ✨', time: '12 min ago' }
  ]);
  const [newComment, setNewComment] = React.useState('');
  const [showStreamLink, setShowStreamLink] = React.useState(false);
  const [streamLink] = React.useState('https://meet.google.com/vtoobe-stream-xyz');
  const [isDescriptionExpanded, setIsDescriptionExpanded] = React.useState(false);

  const creatorNFTs = [
    { id: 1, name: 'Anime Jade', price: '0.5 ETH', img: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/2c0bbc4f6_animejade.png' },
    { id: 2, name: 'Club Diva', price: '0.8 ETH', img: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/0eaa5765e_clubd.png' },
    { id: 3, name: 'Cool Frenchie', price: '1.2 ETH', img: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/895a324dd_coolfrenchie.png' },
    { id: 4, name: 'Cyborg Lion', price: '0.3 ETH', img: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/0f7cb5187_cyborglion.png' },
    { id: 5, name: 'Vtoobe Cat', price: '2.1 ETH', img: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/49d114c97_apple-touch-icon.png' },
  ];
  
  const recommendedVideos = [
    { id: 1, title: '🎌 Tokyo Lofi Beats - Anime Study Mix', creator: 'TokyoBeats', views: '12.3K watching', thumbnail: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/2c0bbc4f6_animejade.png' },
    { id: 2, title: '🇮🇩 Indonesian Folk Songs Karaoke Night', creator: 'JakartaVibes', views: '5.8K watching', thumbnail: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/0eaa5765e_clubd.png' },
    { id: 3, title: '🇵🇭 Filipino Pop Covers with Avatar', creator: 'ManilaMusic', views: '8.9K watching', thumbnail: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/895a324dd_coolfrenchie.png' },
    { id: 4, title: '🍜 Virtual Ramen Cooking Class', creator: 'CyberChef', views: '2.1K watching', thumbnail: 'https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/0f7cb5187_cyborglion.png' },
  ];

  const modes = [
    { icon: MicVocal, label: 'Karaoke Mode' },
    { icon: Church, label: 'Prayer Mode' },
    { icon: Mic, label: 'Public Speaking' },
    { icon: GitFork, label: 'DJ Battles' },
    { icon: Diamond, label: 'NFT Mode' },
    { icon: CookingPot, label: 'Cooking Mode' },
    { icon: Gamepad2, label: 'Gaming Mode' },
  ];

  const specialFeatures = [
    { icon: Gift, label: 'Super Chat Gifts', description: 'Send virtual gifts' },
    { icon: Coins, label: 'Token Rewards', description: 'Earn crypto rewards' },
    { icon: Star, label: 'Fan Badges', description: 'Collect exclusive badges' },
    { icon: Music, label: 'Duet Mode', description: 'Sing together live' },
    { icon: Utensils, label: 'Mukbang Studio', description: 'Virtual dining' },
    { icon: Globe, label: 'Multi-Language', description: 'JP/ID/PH support' },
  ];

  const handleTogglePower = () => setIsTvOn(!isTvOn);

  const handleChangeChannel = (direction) => {
    if (!isTvOn) return;
    setChannel(prev => {
      const newChannel = prev + direction;
      return newChannel < 1 ? 99 : newChannel > 99 ? 1 : newChannel;
    });
    setShowChannel(true);
    setTimeout(() => setShowChannel(false), 1500);
  };

  const handleChangeVolume = (direction) => {
    if (!isTvOn) return;
    setVolume(prev => Math.max(0, Math.min(100, prev + direction * 5)));
  };

  const adjustAntenna = (side) => {
    const adjustment = (Math.random() - 0.5) * 15;
    if (side === 'left') {
      setAntennaLeft(prev => prev + adjustment);
    } else {
      setAntennaRight(prev => prev + adjustment);
    }
  };

  const handleAddComment = (e) => {
    e.preventDefault();
    if (newComment.trim()) {
      const comment = {
        id: Date.now(),
        user: 'You',
        comment: newComment,
        time: 'now'
      };
      setComments(prev => [comment, ...prev]);
      setNewComment('');
    }
  };

  const copyStreamLink = () => {
    navigator.clipboard.writeText(streamLink);
    alert('Stream link copied to clipboard!');
  };

  const nextNft = () => {
    setNftIndex(prev => (prev + 1) % creatorNFTs.length);
  };

  const prevNft = () => {
    setNftIndex(prev => (prev - 1 + creatorNFTs.length) % creatorNFTs.length);
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Orbitron:wght@700&family=Inter:wght@400;500;700&display=swap');
        
        :root {
          --background-color: #0a0a0f;
          --screen-bg: #1a1520;
          --primary-glow: #00e5ff; /* Cyan */
          --secondary-glow: #ff1493; /* Hot Pink */
          --accent-cyan: #00ffff;
          --accent-pink: #ff69b4;
          --warm-brown: #8b4513; 
          --dark-brown: #5d2f1e;
          --text-primary: #ffffff;
          --text-secondary: #e0e0e0;
          --border-color: rgba(0, 229, 255, 0.3);
        }

        body {
          background: linear-gradient(135deg, var(--background-color) 0%, #0f0520 100%);
          color: var(--text-primary);
          font-family: 'Inter', sans-serif;
          overscroll-behavior: none;
        }

        .font-mono { font-family: 'Roboto Mono', monospace; }
        .font-orbitron { font-family: 'Orbitron', sans-serif; }
        
        .tv-casing {
            position: relative;
            width: 100%;
            display: flex;
            flex-direction: column;
            filter: drop-shadow(0 0 30px rgba(0,229,255,0.3));
        }
        
        .tv-screen-container {
            position: relative;
            background: #000;
            border-radius: 2rem;
            overflow: hidden;
            box-shadow: 
              0 0 25px rgba(0, 229, 255, 0.5), /* Cyan glow */
              0 0 50px rgba(255, 20, 147, 0.3), /* Pink glow */
              inset 0 0 20px rgba(0,0,0,1);
            border: 4px solid #333;
            border-bottom: 8px solid #444;
            padding: 2.5%;
            width: 100%;
            aspect-ratio: 16/9;
        }

        .content-wrapper {
            position: relative;
            z-index: 4;
            height: 100%;
            width: 100%;
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1520 100%);
            border-radius: 1rem;
            overflow: hidden;
            transition: opacity 0.5s ease-in-out;
            box-shadow: inset 0 0 80px 20px rgba(0,0,0,0.7);
        }

        .screen-glare {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
              radial-gradient(ellipse 80% 120% at 20% -10%, rgba(0,255,255,0.1), transparent),
              radial-gradient(ellipse 80% 120% at 85% 110%, rgba(255,105,180,0.08), transparent);
            z-index: 5;
            pointer-events: none;
        }
        
        .live-indicator-glow {
          box-shadow: 
            0 0 10px var(--primary-glow), 
            0 0 20px var(--primary-glow),
            0 0 30px var(--secondary-glow);
          animation: pulse-glow 2s infinite alternate;
        }

        @keyframes pulse-glow {
          from { 
            box-shadow: 
              0 0 5px var(--primary-glow), 
              0 0 10px var(--primary-glow),
              0 0 15px var(--secondary-glow);
          }
          to { 
            box-shadow: 
              0 0 15px var(--primary-glow), 
              0 0 25px var(--primary-glow),
              0 0 35px var(--secondary-glow);
          }
        }

        .neon-text {
          text-shadow: 
            0 0 8px var(--primary-glow),
            0 0 15px var(--accent-cyan),
            0 0 20px var(--accent-pink);
        }

        .gradient-bg {
          background: linear-gradient(135deg, var(--primary-glow) 0%, var(--secondary-glow) 50%, var(--accent-pink) 100%);
        }

        .brown-accent {
          background: linear-gradient(135deg, var(--warm-brown) 0%, var(--dark-brown) 100%);
        }
        
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--dark-brown); }
        ::-webkit-scrollbar-thumb {
          background: linear-gradient(var(--primary-glow), var(--accent-pink));
          border-radius: 3px;
        }

        .floating-element {
          animation: float 3s ease-in-out infinite;
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-8px); }
        }

        .tv-antenna {
            position: absolute;
            top: -10px;
            left: 50%;
            width: 80px;
            height: 80px;
            transform: translateX(-50%);
            z-index: -1;
        }
        .antenna-pole {
            position: absolute;
            bottom: 0;
            left: 50%;
            width: 4px;
            height: 100px;
            background: linear-gradient(#e0e0e0, #a0a0a0);
            border-radius: 2px;
            transform-origin: bottom center;
            box-shadow: 1px -1px 3px rgba(0,0,0,0.5);
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        .antenna-pole.left { transform: translateX(-2px) rotate(-25deg); }
        .antenna-pole.right { transform: translateX(-2px) rotate(15deg); }
        .antenna-base {
            position: absolute;
            bottom: -3px;
            left: 50%;
            transform: translateX(-50%);
            width: 20px;
            height: 8px;
            background: #444;
            border-radius: 2px;
            border-bottom: 1px solid #222;
        }
        
        .tv-controls {
            width: 100%;
            min-height: 80px; /* Increased height for NFT carousel */
            background: linear-gradient(135deg, var(--warm-brown), var(--dark-brown));
            border-radius: 0 0 1rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            box-shadow: 0 8px 15px rgba(0,0,0,0.5), inset 0 2px 4px rgba(0,0,0,0.4);
            border: 2px solid #222;
            border-top: none;
        }

        .channel-dial {
            width: 35px;
            height: 35px;
            background: linear-gradient(145deg, #222, #555);
            border-radius: 50%;
            border: 2px solid #111;
            position: relative;
            box-shadow: 0 0 8px rgba(0,0,0,0.7);
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        .channel-dial:hover { transform: scale(1.1) rotate(15deg); }
        .dial-notch {
            position: absolute;
            top: 1px;
            left: 50%;
            transform: translateX(-50%);
            width: 3px;
            height: 7px;
            background: var(--primary-glow); /* Changed from turquoise-glow */
            border-radius: 2px;
            box-shadow: 0 0 4px var(--primary-glow); /* Changed from turquoise-glow */
            transform-origin: center 15.5px;
            transition: transform 0.3s ease;
        }
        
        .control-button {
            width: 40px; /* Increased size */
            height: 40px; /* Increased size */
            background: #333;
            border-radius: 50%;
            border: 2px solid #111;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--secondary-glow); /* Changed from primary-glow */
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .control-button:active, .control-button.active {
            transform: scale(0.95);
            box-shadow: inset 0 3px 6px rgba(0,0,0,0.8), 0 0 10px var(--primary-glow); /* Changed from turquoise-glow */
            color: var(--primary-glow); /* Changed from turquoise-glow */
        }

        .nft-carousel {
          position: relative;
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 1rem;
        }

        .nft-nav-btn {
          background: rgba(0, 229, 255, 0.2); /* Cyan translucent */
          border: 1px solid var(--primary-glow);
          border-radius: 50%;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--primary-glow);
          cursor: pointer;
          transition: all 0.2s ease;
          backdrop-filter: blur(5px);
        }

        .nft-nav-btn:hover {
          background: rgba(0, 229, 255, 0.4);
          box-shadow: 0 0 10px var(--primary-glow);
          transform: scale(1.1);
        }
        
        .nft-card {
          width: 60px; /* Larger NFT card */
          height: 60px; /* Larger NFT card */
          border-radius: 0.75rem;
          border: 2px solid var(--primary-glow);
          overflow: hidden;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 0 15px rgba(0, 229, 255, 0.4); /* Cyan glow */
          margin: 0 0.5rem;
        }
        .nft-card:hover {
          transform: scale(1.1) rotate(3deg);
          box-shadow: 0 0 25px var(--primary-glow), 0 0 35px var(--secondary-glow);
          border-color: var(--secondary-glow);
        }
        
        .vtoobe-brand {
          font-size: 1.5rem;
          font-weight: bold;
          background: linear-gradient(45deg, var(--primary-glow), var(--secondary-glow));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-shadow: 0 0 20px rgba(0, 229, 255, 0.5); /* Cyan glow */
          font-family: 'Orbitron', sans-serif;
          letter-spacing: 2px;
        }

        .tv-static-effect {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAMAAAAp4XiGAAABgVBMVEUAAAD////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////2Uo2GAAAAgXRSTlMAAQIDBAUGCAsOEBESExQVFhcYGRobHB4fICEiJCUnKCkqKywvMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ucHFyc3R1dnd4eXp7fH1+f4CCg4SFh4iJiouMjY6PkJGSlJWWl5iZmpucnZ6foKGio6SlpqeoqaqrrK2ur7CxsrO0tba3uLm6u7y9vr/AwcLDxMXGx8jJysvMzc7P0NHS09TV1tfY2drb3N3e3+Di4uPk5ebn6Onq6+zt7u/w8fLz9PX29/j5+vv8/f7/dlXsywAAAwFJREFUeNqV1fl/FFEcx/FbbVvbtt22bdu2bX/a/m3btm3b1jbttr3dth27Y/f83r3fB84555w7dx/OheCkSZMmjYFQUw4lTpw4mYVw7b0lKTFVITkFlMyYyRJxYhLJeH0sFmNlEivAFSXsZJJIw5sWxBYxUjGz5s1mZplYn8ZisVgsFkUslFDF3GVmsso2s2KxWCyWlM1iMVm8kE0kYslisZgssVgslpL5hYhYLPkFlFksllBlsbAUYqFYLJZwKcXCSlZJLAhWkckiKliYpFEsFAvLhQssKFcYC7FQYWEpFFIsLMsUYrFYLBYLy1IsLAULLBYW5ZKlQhgsLJWYLHYLKFksKJUyhbFRLCxUqVAWC6lYLJaUS8YLiUUpk5YLIWqYWAsFpUIsFCpbLBYSy5kQLkQslGFhsXAxWSlUQLkQKFIsGFhLhYIyYWEhClQslhCFhUslKVYsFhaKhUslpQssKhYuFgsvlZQiFpYLFy6kUqFEsbBYlIWFpUqlUoULKZaFpVCpFCpUvFCFhUuFChdpQoULFypcoQoX3qBCFSpUuEIFC/dsFipcqAIFSpQjQYGKUyBAgQIEilMgQIECBUqU14ACBEpQoECBAgVKUaBAgQIFSlSgQIECBUrUoECBAgVKUaBAgQIFStSgQIECBUrUoECBAgVK0qBAgQIFStagQIECBUrUoECBAgVK0qBAgQIFStagQIECBUrUoECBAgVKUqBAgQIFStSgQIECBUrSoAABAgRKVKBATEB+2UuVCly4QIWKk/7zL106dO+5S5cePZkzc+asWbNmTpw4cSLm//k4c+bMmfXkyZNnzp37+PGjXz59+vLl378/fvz448eP33///ffff//+1Z///gMHDhz+8OHDHx48+PDhwz/cWfzr+P/8h/8BGl/f3bZt2wUAAAAASUVORK5CYII=');
          animation: static-anim 0.08s steps(2, end) infinite;
          z-index: 100;
          pointer-events: none;
        }

        @keyframes static-anim {
          0% { transform: translate(0, 0); }
          100% { transform: translate(1px, 1px); }
        }

        .channel-display {
            position: absolute;
            top: 5%;
            left: 5%;
            font-size: 3rem;
            font-family: 'Orbitron', sans-serif;
            color: rgba(255, 255, 255, 0.9);
            text-shadow: 0 0 8px #fff, 0 0 15px var(--primary-glow); /* Changed from turquoise-glow */
            z-index: 50;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .channel-display.visible { opacity: 1; }
        
        .stream-link-popup {
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: linear-gradient(135deg, var(--warm-brown), var(--dark-brown));
          border: 2px solid var(--primary-glow); /* Changed from primary-glow */
          border-radius: 1rem;
          padding: 2rem;
          box-shadow: 0 0 30px rgba(0,0,0,0.8);
          z-index: 1000;
        }

        .overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0,0,0,0.7);
          backdrop-filter: blur(5px);
          z-index: 999;
        }

        .special-feature-card {
          background: linear-gradient(135deg, rgba(0, 229, 255, 0.1), rgba(255, 105, 180, 0.1)); /* Cyan and Pink translucent background */
          border: 1px solid rgba(0, 229, 255, 0.3); /* Cyan border */
          border-radius: 0.75rem;
          padding: 1rem;
          text-align: center;
          transition: all 0.3s ease;
          cursor: pointer;
        }

        .special-feature-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 10px 25px rgba(0, 229, 255, 0.2); /* Cyan shadow */
          border-color: var(--secondary-glow); /* Pink border on hover */
        }
      `}</style>
      <div className="flex w-full min-h-screen">
        {/* Sidebar */}
        <aside className="w-56 fixed top-0 left-0 h-full brown-accent p-3 flex flex-col border-r-2 border-cyan-500 shadow-xl shadow-cyan-500/20">
          <div className="flex items-center space-x-2 px-2 mb-6 floating-element">
            <div className="w-10 h-10 gradient-bg rounded-full flex items-center justify-center font-bold text-white text-lg font-mono shadow-lg shadow-cyan-500/50">
              <img src="https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/49d114c97_apple-touch-icon.png" alt="Vtoobe Logo" className="w-8 h-8"/>
            </div>
            <span className="vtoobe-brand">Vtoobe</span> {/* Applied new class */}
          </div>
          <nav className="flex flex-col space-y-2">
            <SidebarItem icon={Home} label="Home" active />
            <SidebarItem icon={Tv} label="Browse Streams" />
            <SidebarItem icon={Users} label="Community" />
            <hr className="border-cyan-600/50 my-3"/> {/* Changed from pink */}
            <p className="px-4 text-xs font-bold text-cyan-300 uppercase tracking-wider neon-text">STREAMING MODES</p> {/* Changed from pink */}
            {modes.map((mode, index) => (
              <SidebarItem key={index} icon={mode.icon} label={mode.label} />
            ))}
            <hr className="border-cyan-600/50 my-3"/> {/* Changed from pink */}
            <p className="px-4 text-xs font-bold text-pink-300 uppercase tracking-wider neon-text">SPECIAL FEATURES</p> {/* New section */}
            {specialFeatures.slice(0, 3).map((feature, index) => (
              <SidebarItem key={`special-${index}`} icon={feature.icon} label={feature.label} />
            ))}
          </nav>
           <div className="mt-auto px-4 pt-3 border-t border-cyan-600/30"> {/* Changed from pink */}
            <a href="https://vtoobe.carrd.co" target="_blank" rel="noopener noreferrer" className="text-cyan-300 hover:text-cyan-200 text-sm font-medium transition-colors">About Vtoobe</a> {/* Changed from pink */}
          </div>
        </aside>

        {/* Main Content */}
        <main className="ml-56 flex-1 flex flex-col p-4">
          {/* Header */}
          <header className="flex items-center justify-between p-3 h-16 brown-accent border-b-2 border-cyan-500/50 rounded-xl mb-4"> {/* Changed from pink */}
            <div className="flex items-center gap-3">
                <button className="p-2 text-cyan-300 hover:text-cyan-200 hover:bg-cyan-500/20 rounded-lg transition-all"><Menu size={20}/></button> {/* Changed from pink */}
            </div>
            <div className="flex-1 max-w-lg mx-6">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-cyan-400" /> {/* Changed from pink */}
                <input
                  type="text"
                  placeholder="Search anything..."
                  className="w-full bg-gradient-to-r from-cyan-900/30 to-pink-900/30 border border-cyan-500/50 rounded-full pl-9 pr-3 py-2 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-500/50 transition-all text-cyan-100 placeholder-cyan-300 text-sm" // Changed from pink/red
                />
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <button 
                onClick={() => setShowStreamLink(true)}
                className="flex items-center space-x-2 gradient-bg text-white px-4 py-2 rounded-full font-bold text-sm hover:opacity-90 transition-all shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50" // Changed from pink
              >
                <Video className="w-4 h-4" />
                <span>Go Live</span>
              </button>
              <button className="p-2 rounded-full hover:bg-cyan-500/20 transition-colors border border-cyan-500/30"> {/* Changed from pink */}
                <Bell className="w-4 h-4 text-cyan-300" /> {/* Changed from pink */}
              </button>
              <div className="w-8 h-8 gradient-bg rounded-full flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/40">A</div> {/* Changed from pink */}
            </div>
          </header>

          {/* Content Grid */}
          <div className="flex-1 grid grid-cols-1 xl:grid-cols-4 gap-4">
            <div className="xl:col-span-3 flex flex-col gap-4">
              {/* TV Set */}
              <div className="tv-casing">
                 <div className="tv-antenna">
                    <div className="antenna-pole left" style={{ transform: `translateX(-2px) rotate(${antennaLeft}deg)` }} onClick={() => adjustAntenna('left')}></div>
                    <div className="antenna-pole right" style={{ transform: `translateX(-2px) rotate(${antennaRight}deg)` }} onClick={() => adjustAntenna('right')}></div>
                    <div className="antenna-base"></div>
                </div>
                <div className="tv-screen-container">
                    <div className={`content-wrapper p-1 flex flex-col ${!isTvOn ? 'opacity-20' : ''}`}>
                        {!isTvOn && <div className="tv-static-effect"></div>}
                        <div className={`channel-display ${showChannel ? 'visible' : ''}`}>
                            {String(channel).padStart(2, '0')}
                        </div>
                        <div className="screen-glare"></div>
                        <div className="flex-1 bg-gradient-to-br from-cyan-900/60 via-pink-900/60 to-purple-900/60 rounded-lg flex items-center justify-center relative"> {/* Changed gradient colors */}
                            <div className="text-center floating-element">
                                <div className="w-24 h-24 gradient-bg rounded-full mx-auto mb-4 flex items-center justify-center shadow-xl shadow-cyan-500/50"> {/* Changed shadow */}
                                    <span className="text-4xl">🌸</span>
                                </div>
                                <h3 className="text-2xl font-bold font-mono mb-2 neon-text">Stream Offline</h3>
                                <p className="text-cyan-200">Waiting for VtobeCreator...</p> {/* Changed from pink */}
                                <div className="mt-4 text-pink-300 text-sm"> {/* Added Asian market context */}
                                  <p>🎌 Japanese • 🇮🇩 Indonesian • 🇵🇭 Filipino Content</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="tv-controls">
                    <div className="flex items-center gap-2">
                        <div className={`control-button ${isTvOn ? 'active' : ''}`} onClick={handleTogglePower}>
                            <Power size={18}/> {/* Increased size */}
                        </div>
                        <div className="vtoobe-brand text-sm">VTOOBE</div> {/* New Vtoobe brand text */}
                    </div>
                    
                    {/* NFT Carousel */}
                    <div className="nft-carousel">
                      <button className="nft-nav-btn" onClick={prevNft}>
                        <ChevronLeft size={16} />
                      </button>
                      <div className="nft-card" title={`${creatorNFTs[nftIndex].name} - ${creatorNFTs[nftIndex].price}`}>
                        <img src={creatorNFTs[nftIndex].img} alt={creatorNFTs[nftIndex].name} className="w-full h-full object-cover"/>
                      </div>
                      <button className="nft-nav-btn" onClick={nextNft}>
                        <ChevronRight size={16} />
                      </button>
                    </div>
                    {/* End NFT Carousel */}
                    
                    <div className="flex items-center gap-3 text-center text-cyan-300 font-orbitron"> {/* Changed from pink */}
                        <div className="flex flex-col items-center">
                            <div className="channel-dial" onClick={() => handleChangeVolume(1)}>
                                <div className="dial-notch" style={{transform: `rotate(${volume * 3.6}deg)`}}></div>
                            </div>
                            <label className="mt-1 text-xs">VOL</label>
                        </div>
                         <div className="flex flex-col items-center">
                            <div className="channel-dial" onClick={() => handleChangeChannel(1)}>
                                <div className="dial-notch" style={{transform: `rotate(${channel * 3.6}deg)`}}></div>
                            </div>
                            <label className="mt-1 text-xs">CH</label>
                        </div>
                    </div>
                </div>
              </div>
              
              {/* Stream Info and Description */}
              <div className="brown-accent rounded-xl p-4 border border-cyan-500/30"> {/* Changed from pink */}
                  <h1 className="text-xl font-bold text-cyan-100 mb-2 font-mono neon-text">🎵 Asian Lofi Mix - Study/Relax Beats</h1> {/* Changed title and color */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 gradient-bg rounded-full flex items-center justify-center flex-shrink-0 shadow-lg shadow-cyan-500/40"> {/* Changed shadow */}
                            <span className="font-bold text-white">V</span>
                        </div>
                        <div>
                            <p className="font-bold text-cyan-100">VtobeCreator</p> {/* Changed from pink */}
                            <p className="text-cyan-300 text-sm">15.2K subscribers</p> {/* Changed from pink */}
                        </div>
                        <button className="bg-gradient-to-r from-cyan-600 to-pink-600 hover:from-cyan-700 hover:to-pink-700 px-4 py-2 rounded-full font-bold text-white transition-all shadow-md shadow-cyan-500/30 text-sm"> {/* Changed gradient and shadow */}
                            Subscribe
                        </button>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button className="flex items-center space-x-1.5 brown-accent hover:bg-cyan-500/20 px-3 py-1.5 rounded-full transition-all border border-cyan-500/30 text-cyan-200 text-xs"> {/* Changed from pink */}
                          <Heart className="w-4 h-4"/> <span>3.2K</span>
                      </button>
                      <button className="p-2 brown-accent hover:bg-cyan-500/20 rounded-full transition-all border border-cyan-500/30 text-cyan-200"> {/* Changed from pink */}
                          <ThumbsDown className="w-4 h-4"/>
                      </button>
                      <button className="p-2 brown-accent hover:bg-cyan-500/20 rounded-full transition-all border border-cyan-500/30 text-cyan-200"> {/* Changed from pink */}
                          <Share2 className="w-4 h-4"/>
                      </button>
                       <button className="p-2 brown-accent hover:bg-cyan-500/20 rounded-full transition-all border border-cyan-500/30 text-cyan-200"> {/* Changed from pink */}
                          <Bookmark className="w-4 h-4"/>
                      </button>
                    </div>
                  </div>
                  <div className={`mt-4 bg-black/30 rounded-lg p-3 text-sm transition-all duration-300 ${isDescriptionExpanded ? 'max-h-40' : 'max-h-12 overflow-hidden'}`}>
                    <p className="text-cyan-200 leading-relaxed"> {/* Changed from pink */}
                        🎧 Immerse yourself in soulful Asian-inspired lo-fi beats perfect for studying, working, or relaxing. Features music from Japanese, Indonesian, and Filipino artists with anime-style visuals and NFT collectibles. <br/>
                        <span className="text-pink-300">#lofi #anime #asian #study #vtuber #nft</span> {/* Changed from turquoise */}
                    </p>
                  </div>
                  <button onClick={() => setIsDescriptionExpanded(!isDescriptionExpanded)} className="text-cyan-300 text-xs mt-1 hover:text-white"> {/* Changed from pink */}
                    Show {isDescriptionExpanded ? 'less' : 'more'}
                  </button>
              </div>

              {/* Special Features Grid (New Section) */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                {specialFeatures.slice(3).map((feature, index) => (
                  <div key={`grid-${index}`} className="special-feature-card">
                    <feature.icon className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
                    <h3 className="text-sm font-bold text-cyan-100">{feature.label}</h3>
                    <p className="text-xs text-cyan-300 mt-1">{feature.description}</p>
                  </div>
                ))}
              </div>

              {/* Comments Section */}
              <div className="brown-accent rounded-xl p-4 border border-cyan-500/30"> {/* Changed from pink */}
                <h2 className="text-lg font-bold text-cyan-200 mb-4 flex items-center gap-2"> {/* Changed from pink */}
                  <MessageSquare className="w-5 h-5" />
                  Live Chat
                </h2>
                <div className="space-y-3 mb-4 max-h-60 overflow-y-auto pr-2">
                  {comments.map(comment => (
                    <div key={comment.id} className="bg-black/30 rounded-lg p-2 border border-cyan-500/20"> {/* Changed from pink */}
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-pink-300 font-medium text-sm">{comment.user}</span> {/* Changed from turquoise */}
                        <span className="text-cyan-400 text-xs">{comment.time}</span> {/* Changed from pink */}
                      </div>
                      <p className="text-cyan-100 text-sm">{comment.comment}</p> {/* Changed from pink */}
                    </div>
                  ))}
                </div>
                <form onSubmit={handleAddComment} className="flex gap-2">
                  <input type="text" value={newComment} onChange={(e) => setNewComment(e.target.value)} placeholder="Add a comment..." className="flex-1 bg-black/40 border border-cyan-500/50 rounded-lg px-3 py-2 text-cyan-100 placeholder-cyan-300 text-sm focus:outline-none focus:border-cyan-400"/> {/* Changed from pink */}
                  <button type="submit" className="gradient-bg text-white p-2 rounded-lg hover:opacity-90 transition-opacity"><Send className="w-4 h-4" /></button>
                </form>
              </div>
            </div>

            {/* Recommended Videos */}
            <div className="brown-accent rounded-xl p-4 border border-cyan-500/30"> {/* Changed from pink */}
              <h2 className="text-lg font-bold text-cyan-200 mb-4">Up Next</h2> {/* Changed from pink */}
              <div className="space-y-3">
                {recommendedVideos.map(video => (
                  <div key={video.id} className="flex gap-3 items-center cursor-pointer p-2 rounded-lg hover:bg-cyan-500/10"> {/* Changed from pink */}
                    <img src={video.thumbnail} alt={video.title} className="w-24 h-14 rounded-md object-cover border border-cyan-500/30"/> {/* Used thumbnail and added border */}
                    <div>
                      <h3 className="text-sm font-semibold text-cyan-100 leading-tight">{video.title}</h3> {/* Changed from pink */}
                      <p className="text-xs text-pink-300">{video.creator}</p> {/* Changed from turquoise */}
                      <p className="text-xs text-cyan-300">{video.views}</p> {/* Changed from pink */}
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </main>

        {showStreamLink && (
          <>
            <div className="overlay" onClick={() => setShowStreamLink(false)}></div>
            <div className="stream-link-popup">
              <h3 className="text-xl font-bold text-cyan-200 mb-4 neon-text">Share Your Stream</h3> {/* Changed from pink */}
              <p className="text-cyan-300 mb-4 text-sm">Share this Google Meet link with your followers to invite them to your stream:</p> {/* Changed from pink */}
              <div className="flex items-center gap-2 mb-4">
                <input
                  type="text"
                  value={streamLink}
                  readOnly
                  className="flex-1 bg-black/40 border border-cyan-500/50 rounded-lg px-3 py-2 text-cyan-100 text-sm" // Changed from pink
                />
                <button
                  onClick={copyStreamLink}
                  className="gradient-bg text-white p-2 rounded-lg hover:opacity-90 transition-opacity"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowStreamLink(false)}
                  className="flex-1 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Close
                </button>
                <button
                  onClick={() => {
                    if (navigator.share) {
                      navigator.share({
                        title: 'Join my Vtoobe stream!',
                        text: 'Come watch my live stream on Vtoobe',
                        url: streamLink
                      });
                    } else {
                      copyStreamLink();
                    }
                  }}
                  className="flex-1 gradient-bg text-white px-4 py-2 rounded-lg hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
                >
                  <Share2 className="w-4 h-4" />
                  Share
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
